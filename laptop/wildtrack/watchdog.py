"""Laptop-silence watchdog, shared by the simulators and the Tello link."""

class Watchdog:
    """Same rules as android-bridge Watchdog.kt: silence -> hover -> Virtual Stick off. Each fires once per silence."""

    NONE, HOVER, HANDBACK = "none", "hover", "handback"

    def __init__(self, hover_s: float, handback_s: float):
        self.hover_s = hover_s
        self.handback_s = handback_s
        self.last_cmd_t: float | None = None
        self.hovering = False

    def feed(self, t: float) -> None:
        self.last_cmd_t = t
        self.hovering = False

    def disarm(self) -> None:
        self.last_cmd_t = None
        self.hovering = False

    def check(self, t: float) -> str:
        if self.last_cmd_t is None:
            return self.NONE
        silent = t - self.last_cmd_t
        if silent >= self.handback_s:
            self.disarm()
            return self.HANDBACK
        if silent >= self.hover_s and not self.hovering:
            self.hovering = True
            return self.HOVER
        return self.NONE
