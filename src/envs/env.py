from typing import Any

import gymnasium as gym
import numpy as np


class SIRSEnv(gym.Env):
    def __init__(
        self,
        size: int,
        encounter_rate: float,
        recovery_rate: float,
        resusceptible_rate: float,
        vaccination_rate: float,
        cost_infection: float,
        cost_lockdown: float,
    ):
        self.size = size
        self.encounter_rate = encounter_rate
        self.recovery_rate = recovery_rate
        self.resusceptible_rate = resusceptible_rate
        self.vaccination_rate = vaccination_rate
        self.cost_infection = cost_infection
        self.cost_lockdown = cost_lockdown

        self.uninfected_steps_to_done = 1

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.MultiDiscrete([size + 1, size + 1])
        self._state: np.ndarray = None

        self._terminal_states = [(m_s, 0) for m_s in range(size + 1)]

    def _get_obs(self) -> np.ndarray:
        return self._state

    def _get_info(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "encounter_rate": self.encounter_rate,
            "recovery_rate": self.recovery_rate,
            "resusceptible_rate": self.resusceptible_rate,
            "vaccination_rate": self.vaccination_rate,
            "cost_infection": self.cost_infection,
            "cost_lockdown": self.cost_lockdown,
        }

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[int, int]:
        super().reset(seed=seed)
        if options is not None and "initial_state" in options:
            allowed_states = [
                (m_s, m_i)
                for m_s in range(self.size + 1)
                for m_i in range(self.size + 1)
                if m_s + m_i <= self.size
            ]
            initial_state = tuple(options["initial_state"])
            assert initial_state in allowed_states, (
                f"Invalid initial state {initial_state}"
            )
            self._state = initial_state
        else:
            default_states = [
                (m_s, m_i)
                for m_s in range(self.size + 1)
                for m_i in range(self.size + 1)
                if m_s + m_i <= self.size
            ]
            self._state = default_states[np.random.choice(len(default_states))]
        return self._get_obs(), self._get_info()

    def _cost_fn(self, state: tuple[int, int], action: int) -> float:
        m_s, m_i = state
        cost = (self.cost_lockdown - action) * (
            m_s / self.size
        ) + self.cost_infection * (m_i / self.size)
        return cost

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, bool, Any]:
        m_s, m_i = self._state
        unif = 1 / (
            self.size
            * (
                self.encounter_rate
                + self.recovery_rate
                + self.resusceptible_rate
                + self.vaccination_rate
            )
        )
        w_I = unif * self.encounter_rate * action * m_s * m_i / self.size
        w_V = unif * self.vaccination_rate * m_s
        w_R = unif * self.recovery_rate * m_i
        w_S = unif * self.resusceptible_rate * (self.size - m_s - m_i)
        w_hat = 1 - w_V - w_I - w_R - w_S

        possible_next_states = [
            (m_s - 1, m_i + 1),
            (m_s - 1, m_i),
            (m_s, m_i - 1),
            (m_s + 1, m_i),
            (m_s, m_i),
        ]
        next_state = possible_next_states[
            np.random.choice(
                len(possible_next_states),
                p=[w_I, w_V, w_R, w_S, w_hat],
            )
        ]
        reward = -self._cost_fn(self._state, action)
        self._state = next_state
        return np.array(self._state), reward, False, False, self._get_info()
