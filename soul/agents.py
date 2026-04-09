"""
Subconscious Agents — Lin Xia's Background Processing Minds

Four specialized agents running as async tasks, communicating via the
SubconsciousBus. Each handles a specific aspect of Lin Xia's "inner life":

  1. MemoryAgent:     Background memory retrieval + consolidation
  2. EmotionAnalyst:  Real-time emotion analysis (text + physiological)
  3. MetabolismAgent: Bio-clock, sleep cycle, dream generation
  4. CompanionAgent:  Proactive outreach decision-making

Each agent:
  - Runs in its own asyncio.Task
  - Publishes results to the bus
  - Subscribes to relevant events from other agents
  - Has its own polling interval
"""

import asyncio
import time
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soul.subconscious_bus import SubconsciousBus, BusEvent


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """Base class for all subconscious agents."""

    def __init__(self, name: str, bus: SubconsciousBus,
                 poll_interval: float = 5.0):
        self.name = name
        self.bus = bus
        self.poll_interval = poll_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Shared state (updated by bus events)
        self.shared_state: dict = {}

    async def setup(self):
        """Override: subscribe to relevant events."""
        pass

    async def tick(self):
        """Override: called every poll_interval seconds."""
        pass

    async def run(self):
        """Main loop — calls tick() periodically."""
        self._running = True
        await self.setup()
        while self._running:
            try:
                await self.tick()
            except Exception as e:
                print(f"  [{self.name}] Error in tick: {e}")
            await asyncio.sleep(self.poll_interval)

    def start(self, loop: asyncio.AbstractEventLoop = None):
        """Start agent as a background task."""
        if loop:
            self._task = loop.create_task(self.run())
        else:
            self._task = asyncio.create_task(self.run())
        return self._task

    def stop(self):
        """Stop the agent."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def publish(self, topic: str, payload: dict = None):
        """Convenience: publish an event."""
        await self.bus.publish(BusEvent(
            topic=topic,
            payload=payload or {},
            source=self.name,
        ))


# ---------------------------------------------------------------------------
# 1. Memory Agent
# ---------------------------------------------------------------------------

class MemoryAgent(BaseAgent):
    """
    Handles background memory operations:
      - Pre-fetches relevant memories based on recent conversation
      - Consolidates short-term memories during "sleep"
      - Publishes retrieved memories for other agents
    """

    def __init__(self, bus: SubconsciousBus,
                 memory_store=None, memory_retriever=None,
                 memory_consolidator=None):
        super().__init__("MemoryAgent", bus, poll_interval=2.0)
        self.memory_store = memory_store
        self.memory_retriever = memory_retriever
        self.consolidator = memory_consolidator

        self.last_query = ""
        self.cached_memories: list[str] = []

    async def setup(self):
        # Listen for user input events to proactively retrieve
        self.bus.subscribe("user.input", self._on_user_input, self.name)
        self.bus.subscribe("bio.sleep_start", self._on_sleep, self.name)

    async def _on_user_input(self, event: BusEvent):
        """When user sends a message, proactively retrieve memories."""
        query = event.payload.get("text", "")
        if query and self.memory_retriever:
            self.last_query = query
            self.cached_memories = self.memory_retriever.retrieve(query, n_results=3)
            await self.publish("memory.retrieved", {
                "memories": self.cached_memories,
                "query": query,
                "count": len(self.cached_memories),
            })

    async def _on_sleep(self, event: BusEvent):
        """When metabolism agent triggers sleep, consolidate memories."""
        if self.consolidator:
            stats = self.consolidator.consolidate(max_memories=20)
            await self.publish("memory.consolidated", {
                "processed": stats.get("processed", 0),
                "new_entities": stats.get("new_entities", 0),
                "new_relations": stats.get("new_relations", 0),
            })

    async def tick(self):
        """Periodic: report memory stats."""
        if self.memory_store:
            count = self.memory_store.get_memory_count()
            if count > 0:
                await self.publish("memory.stats", {"count": count})


# ---------------------------------------------------------------------------
# 2. Emotion Analyst Agent
# ---------------------------------------------------------------------------

class EmotionAnalystAgent(BaseAgent):
    """
    Analyzes user emotions from multiple signals:
      - Text sentiment (from conversation)
      - Physiological signals (from health perception)
      - Behavioral patterns (response time, message length)
    
    Publishes a unified emotional assessment for the cognitive graph.
    """

    def __init__(self, bus: SubconsciousBus,
                 health_perception=None):
        super().__init__("EmotionAnalyst", bus, poll_interval=3.0)
        self.health = health_perception

        # Tracked state
        self.user_mood = "neutral"
        self.user_stress = 0.0
        self.user_bpm = 0
        self.text_sentiment = "neutral"
        self.last_message_length = 0
        self.message_frequency = 0.0  # messages per minute

    async def setup(self):
        self.bus.subscribe("user.input", self._on_user_input, self.name)
        self.bus.subscribe("health.updated", self._on_health, self.name)

    async def _on_user_input(self, event: BusEvent):
        """Analyze text sentiment from user input."""
        text = event.payload.get("text", "")
        self.last_message_length = len(text)

        # Simple keyword-based sentiment
        positive_words = ["好", "开心", "喜欢", "谢谢", "爱", "棒", "哈哈", "嘻嘻"]
        negative_words = ["烦", "累", "难过", "讨厌", "生气", "不想", "无聊", "垃圾"]
        aggressive_words = ["滚", "闭嘴", "AI", "人工智能", "假的", "工具"]

        pos = sum(1 for w in positive_words if w in text)
        neg = sum(1 for w in negative_words if w in text)
        agg = sum(1 for w in aggressive_words if w in text)

        if agg > 0:
            self.text_sentiment = "hostile"
            self.user_mood = "angry"
        elif neg > pos:
            self.text_sentiment = "negative"
            self.user_mood = "sad"
        elif pos > neg:
            self.text_sentiment = "positive"
            self.user_mood = "happy"
        else:
            self.text_sentiment = "neutral"
            self.user_mood = "neutral"

        await self._publish_assessment()

    async def _on_health(self, event: BusEvent):
        """Update physiological signals."""
        self.user_bpm = event.payload.get("bpm", 0)
        self.user_stress = event.payload.get("stress", 0.0)

    async def _publish_assessment(self):
        """Publish unified emotional assessment."""
        await self.publish("emotion.updated", {
            "user_mood": self.user_mood,
            "text_sentiment": self.text_sentiment,
            "user_stress": self.user_stress,
            "user_bpm": self.user_bpm,
            "message_length": self.last_message_length,
        })

    async def tick(self):
        """Periodic health polling."""
        if self.health and self.health.last_state.status == "tracking":
            self.user_bpm = self.health.last_state.bpm
            self.user_stress = self.health.last_state.stress_level
            await self.publish("health.updated", {
                "bpm": self.user_bpm,
                "stress": self.user_stress,
                "status": self.health.last_state.status,
            })


# ---------------------------------------------------------------------------
# 3. Metabolism Agent
# ---------------------------------------------------------------------------

class MetabolismAgent(BaseAgent):
    """
    Manages Lin Xia's biological rhythm:
      - Updates bio-clock state every minute
      - Triggers sleep/wake transitions
      - Generates dreams during sleep
      - Tracks loneliness based on chat absence
    """

    def __init__(self, bus: SubconsciousBus,
                 bio_clock=None, dream_engine=None):
        super().__init__("MetabolismAgent", bus, poll_interval=60.0)
        self.bio_clock = bio_clock
        self.dream_engine = dream_engine

        self.last_chat_time = time.time()
        self.was_sleeping = False
        self.dream_text = ""

    async def setup(self):
        self.bus.subscribe("user.input", self._on_user_input, self.name)

    async def _on_user_input(self, event: BusEvent):
        """Track last chat time for loneliness calculation."""
        self.last_chat_time = time.time()

    async def tick(self):
        """Update bio state every minute."""
        if not self.bio_clock:
            return

        state = self.bio_clock.get_state(last_chat_time=self.last_chat_time)

        # Detect sleep transition
        is_sleeping = state.is_sleeping
        if is_sleeping and not self.was_sleeping:
            # Just fell asleep
            await self.publish("bio.sleep_start", {
                "time": time.time(),
            })
            # Generate dream
            if self.dream_engine:
                self.dream_text = self.dream_engine.dream()
                await self.publish("bio.dream_ready", {
                    "dream": self.dream_text,
                })

        elif not is_sleeping and self.was_sleeping:
            # Just woke up
            await self.publish("bio.wake_up", {
                "dream": self.dream_text,
            })

        self.was_sleeping = is_sleeping

        # Publish state update
        await self.publish("bio.state_changed", {
            "time_of_day": state.time_of_day,
            "energy": state.energy,
            "is_sleeping": state.is_sleeping,
            "loneliness": state.loneliness,
            "hours_since_chat": state.hours_since_last_chat,
        })


# ---------------------------------------------------------------------------
# 4. Companion Agent
# ---------------------------------------------------------------------------

class CompanionAgent(BaseAgent):
    """
    Decides when and how Lin Xia should proactively reach out.
    Considers:
      - Time since last chat (loneliness)
      - User's emotional state
      - Current screen context
      - Relationship stage
    """

    def __init__(self, bus: SubconsciousBus,
                 companion_engine=None, proactive_engine=None,
                 relationship=None):
        super().__init__("CompanionAgent", bus, poll_interval=300.0)  # 5 min
        self.companion = companion_engine
        self.proactive = proactive_engine
        self.relationship = relationship

        # State from other agents
        self.user_mood = "neutral"
        self.hours_since_chat = 0.0
        self.current_app = ""
        self.loneliness = 0.0

    async def setup(self):
        self.bus.subscribe("emotion.updated", self._on_emotion, self.name)
        self.bus.subscribe("bio.state_changed", self._on_bio, self.name)

    async def _on_emotion(self, event: BusEvent):
        self.user_mood = event.payload.get("user_mood", "neutral")

    async def _on_bio(self, event: BusEvent):
        self.hours_since_chat = event.payload.get("hours_since_chat", 0)
        self.loneliness = event.payload.get("loneliness", 0)

    async def tick(self):
        """Decide if Lin Xia should proactively reach out."""
        if not self.companion:
            return

        # Check proactive interval based on relationship
        if self.relationship:
            interval = self.relationship.get_proactive_interval()
            if self.hours_since_chat < interval:
                return  # Not time yet

        # Use companion engine to decide
        action = self.companion.decide_action(
            screen_app=self.current_app,
            user_mood=self.user_mood,
            hours_since_chat=self.hours_since_chat,
        )

        if action:
            await self.publish("companion.action_ready", {
                "activity": action.activity.value,
                "message": action.message,
                "priority": action.priority,
                "context_for_llm": action.context_for_llm,
            })


# ---------------------------------------------------------------------------
# Agent Orchestra — Manages all agents
# ---------------------------------------------------------------------------

class AgentOrchestra:
    """
    Manages the lifecycle of all subconscious agents.
    Provides a simple interface for the main app to start/stop/query agents.
    """

    def __init__(self, bus: SubconsciousBus):
        self.bus = bus
        self.agents: dict[str, BaseAgent] = {}
        self._started = False

    def register(self, agent: BaseAgent):
        """Register an agent with the orchestra."""
        self.agents[agent.name] = agent

    def start_all(self):
        """Start all agents and the bus."""
        for agent in self.agents.values():
            agent.start()
        self.bus.start_background()
        self._started = True
        print(f"[ORCHESTRA] Started {len(self.agents)} agents + bus")

    def stop_all(self):
        """Stop all agents and the bus."""
        for agent in self.agents.values():
            agent.stop()
        self.bus.stop()
        self._started = False
        print("[ORCHESTRA] All agents stopped")

    def get_status(self) -> dict:
        """Get status of all agents and the bus."""
        return {
            "running": self._started,
            "agents": list(self.agents.keys()),
            "bus_stats": self.bus.get_stats(),
            "recent_events": self.bus.get_recent_events(5),
        }

    async def broadcast_user_input(self, text: str):
        """Broadcast a user input event to all agents."""
        await self.bus.publish(BusEvent(
            topic="user.input",
            payload={"text": text, "timestamp": time.time()},
            source="MainApp",
        ))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def demo():
    """Demo all agents working together."""
    print("=" * 60)
    print("Subconscious Agents — Lin Xia's Inner Minds")
    print("=" * 60)

    bus = SubconsciousBus(debug=True)

    # Create agents with mock dependencies
    memory_agent = MemoryAgent(bus)
    emotion_agent = EmotionAnalystAgent(bus)
    metabolism_agent = MetabolismAgent(bus)
    metabolism_agent.poll_interval = 2.0  # Speed up for demo
    companion_agent = CompanionAgent(bus)
    companion_agent.poll_interval = 3.0

    # Create orchestra
    orchestra = AgentOrchestra(bus)
    orchestra.register(memory_agent)
    orchestra.register(emotion_agent)
    orchestra.register(metabolism_agent)
    orchestra.register(companion_agent)

    # Start all
    print("\n[1] Starting agents...")
    bus_task = asyncio.create_task(bus.start())

    for agent in orchestra.agents.values():
        agent.start()

    await asyncio.sleep(0.5)

    # Simulate user input
    print("\n[2] Simulating user interactions...")
    await orchestra.broadcast_user_input("你好！今天过得怎么样？")
    await asyncio.sleep(1.0)

    await orchestra.broadcast_user_input("我今天工作好累啊...")
    await asyncio.sleep(1.0)

    await orchestra.broadcast_user_input("你就是个人工智能而已")
    await asyncio.sleep(1.0)

    # Check status
    print(f"\n[3] Orchestra Status")
    status = orchestra.get_status()
    print(f"  Running: {status['running']}")
    print(f"  Agents: {', '.join(status['agents'])}")
    print(f"  Bus: published={status['bus_stats']['published']}, "
          f"delivered={status['bus_stats']['delivered']}")

    print(f"\n[4] Recent Events")
    for evt in bus.get_recent_events(10):
        print(f"  {evt['source']:20s} → {evt['topic']}")

    # Cleanup
    orchestra.stop_all()
    bus.stop()
    bus_task.cancel()
    try:
        await bus_task
    except asyncio.CancelledError:
        pass

    print(f"\n{'='*60}")
    print(f"  ✅ Demo complete. {bus.events_published} events published, "
          f"{bus.events_delivered} delivered.")


if __name__ == "__main__":
    asyncio.run(demo())
