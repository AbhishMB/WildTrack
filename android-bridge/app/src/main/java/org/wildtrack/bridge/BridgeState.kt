package org.wildtrack.bridge

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** What the phone screen shows. Written by the service, read by MainActivity. */
data class Status(
    val sdk: String = "starting",
    val registered: Boolean = false,
    val product: String? = null,
    val laptop: String? = null,
    val vsEnabled: Boolean = false,
    val hovering: Boolean = false,
    val rtspUrl: String? = null,
    val streaming: Boolean = false,
    val streamInfo: String = "",
    val simulator: Boolean = false,
    val axis: String = AxisMapping().describe(),
    val telemetry: TelemetrySnapshot = TelemetrySnapshot(),
    val lastCommandMs: Long? = null,
    val log: List<String> = emptyList(),
)

object BridgeState {
    private val _status = MutableStateFlow(Status())
    val status: StateFlow<Status> = _status

    fun update(f: (Status) -> Status) = _status.update(f)

    fun log(line: String) {
        val stamp = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
        _status.update { it.copy(log = (listOf("$stamp $line") + it.log).take(40)) }
    }
}
