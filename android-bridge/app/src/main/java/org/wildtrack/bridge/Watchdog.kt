package org.wildtrack.bridge

/**
 * Laptop silence detector (docs/SPEC.md, safety table). Pure: the caller passes the clock.
 *
 *   no command for 300 ms -> HOVER    (send zero velocity until commands resume)
 *   no command for 3 s    -> HANDBACK (Virtual Stick off, alarm; the pilot flies)
 *
 * Each action fires once per silence. Disarmed until the first command after enabling,
 * and again after a hand-back. The laptop simulator has the same rules (sim/drone.py).
 */
class Watchdog(private val hoverMs: Long, private val handbackMs: Long) {
    enum class Action { NONE, HOVER, HANDBACK }

    private var lastCommandMs: Long? = null

    var hovering = false
        private set

    val armed: Boolean get() = lastCommandMs != null

    @Synchronized
    fun feed(nowMs: Long) {
        lastCommandMs = nowMs
        hovering = false
    }

    @Synchronized
    fun disarm() {
        lastCommandMs = null
        hovering = false
    }

    @Synchronized
    fun check(nowMs: Long): Action {
        val last = lastCommandMs ?: return Action.NONE
        val silent = nowMs - last
        if (silent >= handbackMs) {
            disarm()
            return Action.HANDBACK
        }
        if (silent >= hoverMs && !hovering) {
            hovering = true
            return Action.HOVER
        }
        return Action.NONE
    }
}
