from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_checkpoint_roundtrip_if_torch_available():
    try:
        from mountaincar.rl.checkpoints import load_warmstart_agent, save_agent_checkpoint
        from mountaincar.rl.agent import DQNAgent, DQNConfig
    except Exception:
        return

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "agent.pt"
        agent = DQNAgent(DQNConfig(), device="cpu")
        save_agent_checkpoint(agent, ckpt, extra={"tag": "roundtrip"})
        loaded = load_warmstart_agent(ckpt, device="cpu")
        assert isinstance(loaded, DQNAgent)
        assert ckpt.exists()
