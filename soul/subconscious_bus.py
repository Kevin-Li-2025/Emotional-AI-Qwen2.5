"""
Subconscious Bus — Event-driven Async Communication for Lin Xia's Agents

A lightweight pub/sub event bus that allows Lin Xia's specialized sub-agents
(Memory, Emotion, Metabolism, Companion) to communicate asynchronously.

Architecture:
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Memory   │  │ Emotion  │  │ Metab.   │  │ Companion│
  │ Agent    │  │ Analyst  │  │ Agent    │  │ Agent    │
  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │              │
       └──────────┬───┴──────────┬───┘              │
                  ▼              ▼                   │
          ┌──────────────────────────────┐           │
          │    SubconsciousBus           │◄──────────┘
          │    (asyncio.Queue based)     │
          └──────────────────────────────┘

Events are typed with a topic string (e.g., "memory.retrieved", "bio.state_changed")
and carry a dict payload. Subscribers filter by topic prefix.
"""

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Awaitable
from collections import defaultdict


@dataclass
class BusEvent:
    """A single event on the subconscious bus."""
    topic: str                    # e.g., "memory.retrieved", "bio.dream_ready"
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""              # Agent name that published this


@dataclass
class Subscription:
    """A subscriber registered on the bus."""
    topic_prefix: str             # e.g., "memory." subscribes to all memory events
    callback: Callable[..., Awaitable]  # async def handler(event: BusEvent)
    subscriber_name: str = ""


class SubconsciousBus:
    """
    Async event bus for inter-agent communication.
    
    Usage:
        bus = SubconsciousBus()
        
        # Subscribe
        bus.subscribe("memory.", callback_fn, "EmotionAgent")
        
        # Publish
        await bus.publish(BusEvent(topic="memory.retrieved", payload={...}))
        
        # Start bus processing
        await bus.start()
    """

    def __init__(self, max_queue_size: int = 100, debug: bool = False):
        self.queue: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=max_queue_size)
        self.subscriptions: list[Subscription] = []
        self.event_log: list[dict] = []
        self.debug = debug
        self._running = False
        self._task: asyncio.Task | None = None

        # Metrics
        self.events_published = 0
        self.events_delivered = 0
        self.errors = 0

    def subscribe(self, topic_prefix: str,
                  callback: Callable[..., Awaitable],
                  subscriber_name: str = ""):
        """
        Register a subscriber for events matching the topic prefix.
        
        Examples:
            bus.subscribe("memory.", handler)     # All memory events
            bus.subscribe("bio.dream", handler)   # Only dream events
            bus.subscribe("", handler)             # ALL events
        """
        sub = Subscription(
            topic_prefix=topic_prefix,
            callback=callback,
            subscriber_name=subscriber_name,
        )
        self.subscriptions.append(sub)
        if self.debug:
            print(f"  [BUS] {subscriber_name} subscribed to '{topic_prefix}*'")

    async def publish(self, event: BusEvent):
        """
        Publish an event to the bus.
        Non-blocking if queue is not full (drops event if full).
        """
        try:
            self.queue.put_nowait(event)
            self.events_published += 1
            if self.debug:
                print(f"  [BUS] ← {event.source}: {event.topic} "
                      f"(payload: {len(event.payload)} keys)")
        except asyncio.QueueFull:
            if self.debug:
                print(f"  [BUS] ⚠ Queue full, dropping: {event.topic}")

    async def _dispatch(self, event: BusEvent):
        """Route an event to all matching subscribers."""
        for sub in self.subscriptions:
            if event.topic.startswith(sub.topic_prefix):
                try:
                    await sub.callback(event)
                    self.events_delivered += 1
                except Exception as e:
                    self.errors += 1
                    if self.debug:
                        print(f"  [BUS] ❌ Error in {sub.subscriber_name}: {e}")
                        traceback.print_exc()

        # Log event
        self.event_log.append({
            "topic": event.topic,
            "source": event.source,
            "timestamp": event.timestamp,
            "payload_keys": list(event.payload.keys()),
        })
        # Keep log bounded
        if len(self.event_log) > 500:
            self.event_log = self.event_log[-200:]

    async def start(self):
        """Start the event processing loop."""
        self._running = True
        if self.debug:
            print("  [BUS] Started event processing loop")

        while self._running:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue  # No events, keep polling
            except asyncio.CancelledError:
                break

    def stop(self):
        """Stop the event processing loop."""
        self._running = False
        if self._task:
            self._task.cancel()

    def start_background(self, loop: asyncio.AbstractEventLoop = None):
        """Start the bus in the background as an asyncio task."""
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()

        self._task = loop.create_task(self.start())
        return self._task

    def get_stats(self) -> dict:
        """Get bus statistics."""
        return {
            "published": self.events_published,
            "delivered": self.events_delivered,
            "errors": self.errors,
            "subscribers": len(self.subscriptions),
            "queue_size": self.queue.qsize(),
            "log_size": len(self.event_log),
        }

    def get_recent_events(self, n: int = 10) -> list[dict]:
        """Get the N most recent events for debugging."""
        return self.event_log[-n:]


# ---------------------------------------------------------------------------
# Convenience: Synchronous wrapper for non-async contexts
# ---------------------------------------------------------------------------

class SyncBusAdapter:
    """
    Synchronous adapter for the SubconsciousBus.
    Useful for integrating into non-async code (like Gradio callbacks).
    """

    def __init__(self, bus: SubconsciousBus):
        self.bus = bus
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = None

    def start(self):
        """Start the bus in a background thread."""
        import threading

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self.bus.start())

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

    def publish_sync(self, topic: str, payload: dict = None, source: str = ""):
        """Publish an event synchronously (thread-safe)."""
        event = BusEvent(topic=topic, payload=payload or {}, source=source)
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self.bus.publish(event), self._loop
            )

    def stop(self):
        """Stop the bus."""
        self.bus.stop()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def demo():
    """Demo the subconscious bus with mock agents."""
    print("=" * 60)
    print("Subconscious Bus — Event-Driven Agent Communication")
    print("=" * 60)

    bus = SubconsciousBus(debug=True)

    # Track received events
    received = []

    async def memory_handler(event: BusEvent):
        received.append(("MemoryAgent", event.topic))
        print(f"    → MemoryAgent received: {event.topic}")

    async def emotion_handler(event: BusEvent):
        received.append(("EmotionAgent", event.topic))
        print(f"    → EmotionAgent received: {event.topic}")

    async def companion_handler(event: BusEvent):
        received.append(("CompanionAgent", event.topic))
        print(f"    → CompanionAgent received: {event.topic}")

    # Subscribe
    bus.subscribe("memory.", memory_handler, "MemoryAgent")
    bus.subscribe("emotion.", emotion_handler, "EmotionAgent")
    bus.subscribe("bio.", companion_handler, "CompanionAgent")
    bus.subscribe("", emotion_handler, "EmotionAgent (all)")  # Listen to everything

    # Start bus
    bus_task = asyncio.create_task(bus.start())

    # Give the bus time to start
    await asyncio.sleep(0.1)

    # Publish test events
    print("\n[1] Publishing events...")

    await bus.publish(BusEvent(
        topic="memory.retrieved",
        payload={"memories": ["用户喜欢薰衣草"], "count": 1},
        source="MemoryAgent",
    ))

    await bus.publish(BusEvent(
        topic="emotion.updated",
        payload={"mood": "happy", "intensity": 7},
        source="EmotionAnalyst",
    ))

    await bus.publish(BusEvent(
        topic="bio.state_changed",
        payload={"energy": 0.8, "time_of_day": "evening"},
        source="MetabolismAgent",
    ))

    await bus.publish(BusEvent(
        topic="companion.action_ready",
        payload={"action": "check_in", "message": "你在忙什么呀？"},
        source="CompanionAgent",
    ))

    # Wait for processing
    await asyncio.sleep(0.5)

    # Stop
    bus.stop()
    bus_task.cancel()
    try:
        await bus_task
    except asyncio.CancelledError:
        pass

    # Results
    print(f"\n[2] Results")
    stats = bus.get_stats()
    print(f"  Published:  {stats['published']} events")
    print(f"  Delivered:  {stats['delivered']} events")
    print(f"  Errors:     {stats['errors']}")
    print(f"  Subscriptions: {stats['subscribers']}")

    print(f"\n[3] Event Log")
    for e in bus.get_recent_events():
        print(f"  {e['source']:20s} → {e['topic']}")

    # Verify
    print(f"\n[4] Verification")
    expected_deliveries = 8  # 4 events × varying subscriber matches
    actual = len(received)
    print(f"  Expected ≥ 4 deliveries, got {actual}: "
          f"{'✅ PASS' if actual >= 4 else '❌ FAIL'}")


if __name__ == "__main__":
    asyncio.run(demo())
