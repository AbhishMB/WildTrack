package org.wildtrack.bridge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Same cases as the laptop's tests/test_sim_closed_loop.py::test_watchdog_rules. */
class WatchdogTest {
    private val w = Watchdog(300, 3000)

    @Test
    fun disarmedUntilFirstCommand() {
        assertEquals(Watchdog.Action.NONE, w.check(10_000))
        assertFalse(w.armed)
    }

    @Test
    fun hoverAfter300msThenHandbackAfter3s() {
        w.feed(0)
        assertEquals(Watchdog.Action.NONE, w.check(299))
        assertEquals(Watchdog.Action.HOVER, w.check(300))
        assertTrue(w.hovering)
        assertEquals(Watchdog.Action.NONE, w.check(500)) // once per silence
        assertEquals(Watchdog.Action.HANDBACK, w.check(3000))
        assertFalse(w.armed)
        assertEquals(Watchdog.Action.NONE, w.check(9000)) // disarmed after hand-back
    }

    @Test
    fun commandsResumeEndsHover() {
        w.feed(0)
        assertEquals(Watchdog.Action.HOVER, w.check(400))
        w.feed(450)
        assertFalse(w.hovering)
        assertEquals(Watchdog.Action.NONE, w.check(700))
        assertEquals(Watchdog.Action.HOVER, w.check(750)) // a new silence hovers again
    }

    @Test
    fun steadyTenHzNeverFires() {
        for (t in 0L..60_000L step 100) {
            w.feed(t)
            assertEquals(Watchdog.Action.NONE, w.check(t + 99))
        }
    }
}
