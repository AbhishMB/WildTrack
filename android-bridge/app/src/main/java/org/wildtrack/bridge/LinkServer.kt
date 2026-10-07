package org.wildtrack.bridge

import io.ktor.server.application.install
import io.ktor.server.cio.CIO
import io.ktor.server.engine.ApplicationEngine
import io.ktor.server.engine.embeddedServer
import io.ktor.server.plugins.origin
import io.ktor.server.routing.routing
import io.ktor.server.websocket.DefaultWebSocketServerSession
import io.ktor.server.websocket.WebSockets
import io.ktor.server.websocket.pingPeriod
import io.ktor.server.websocket.timeout
import io.ktor.server.websocket.webSocket
import io.ktor.websocket.CloseReason
import io.ktor.websocket.Frame
import io.ktor.websocket.close
import io.ktor.websocket.readText
import java.time.Duration

/**
 * WebSocket server for the laptop: telemetry and events out, commands in.
 * One laptop at a time; a new connection replaces the old one (a laptop that
 * reconnects after a Wi-Fi drop must not be locked out by its own dead socket).
 */
class LinkServer(
    private val port: Int,
    private val onCommand: (Command) -> Unit,
    private val onClient: (connected: Boolean, host: String?) -> Unit,
) {
    private var engine: ApplicationEngine? = null

    @Volatile private var session: DefaultWebSocketServerSession? = null

    val connected: Boolean get() = session != null

    fun start() {
        engine = embeddedServer(CIO, port = port, host = "0.0.0.0") {
            install(WebSockets) {
                pingPeriod = Duration.ofSeconds(1)
                timeout = Duration.ofSeconds(3)
                maxFrameSize = 1L shl 16
                masking = false
            }
            routing {
                webSocket(BridgeConfig.WS_PATH) {
                    session?.close(CloseReason(CloseReason.Codes.GOING_AWAY, "replaced by a new laptop connection"))
                    session = this
                    val host = call.request.origin.remoteHost
                    onClient(true, host)
                    try {
                        for (frame in incoming) {
                            if (frame !is Frame.Text) continue
                            val cmd = try {
                                Protocol.parse(frame.readText())
                            } catch (e: Exception) {
                                BridgeState.log("bad message from laptop: ${e.message}")
                                null
                            }
                            if (cmd != null) onCommand(cmd)
                        }
                    } finally {
                        if (session === this) {
                            session = null
                            onClient(false, host)
                        }
                    }
                }
            }
        }.start(wait = false)
    }

    /** Non-blocking; drops the message if the socket is backed up (telemetry supersedes itself). */
    fun send(text: String) {
        session?.outgoing?.trySend(Frame.Text(text))
    }

    fun stop() {
        engine?.stop(500, 1000)
        engine = null
    }
}
