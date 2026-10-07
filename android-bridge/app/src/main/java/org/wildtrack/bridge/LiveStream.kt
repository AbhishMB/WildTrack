package org.wildtrack.bridge

import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.datacenter.livestream.LiveStreamSettings
import dji.v5.manager.datacenter.livestream.LiveStreamStatus
import dji.v5.manager.datacenter.livestream.LiveStreamStatusListener
import dji.v5.manager.datacenter.livestream.LiveStreamType
import dji.v5.manager.datacenter.livestream.LiveVideoBitrateMode
import dji.v5.manager.datacenter.livestream.StreamQuality
import dji.v5.manager.datacenter.livestream.settings.RtspSettings
import java.net.Inet4Address
import java.net.NetworkInterface

/**
 * RTSP server on the phone via MSDK's LiveStreamManager; the laptop pulls
 * rtsp://aaa:aaa@<phone-ip>:8554/streaming/live/1 with PyAV.
 */
class LiveStream(private val fullHd: () -> Boolean) {
    private val lsm get() = MediaDataCenter.getInstance().liveStreamManager

    private val status = object : LiveStreamStatusListener {
        override fun onLiveStreamStatusUpdate(s: LiveStreamStatus?) {
            if (s == null) return
            val res = s.resolution?.let { "${it.width}x${it.height}" } ?: "?"
            BridgeState.update { it.copy(streaming = s.isStreaming, streamInfo = "$res ${s.fps} fps, rtt ${s.rtt} ms") }
        }

        override fun onError(error: IDJIError?) {
            BridgeState.log("RTSP error: ${error?.description()}")
        }
    }

    fun start() {
        val rtsp = RtspSettings.Builder()
            .setUserName(BridgeConfig.RTSP_USER)
            .setPassWord(BridgeConfig.RTSP_PASSWORD)
            .setPort(BridgeConfig.RTSP_PORT)
            .build()
        lsm.setLiveStreamSettings(LiveStreamSettings.Builder().setLiveStreamType(LiveStreamType.RTSP).setRtspSettings(rtsp).build())
        lsm.setCameraIndex(ComponentIndexType.LEFT_OR_MAIN)
        // HD (1280 px) is what the spec's small-object numbers assume; FULL_HD trades delay for detail.
        lsm.setLiveStreamQuality(if (fullHd()) StreamQuality.FULL_HD else StreamQuality.HD)
        lsm.setLiveVideoBitrateMode(LiveVideoBitrateMode.AUTO)
        lsm.addLiveStreamStatusListener(status)
        lsm.startStream(object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                val url = url()
                BridgeState.update { it.copy(rtspUrl = url) }
                BridgeState.log("RTSP: $url")
            }

            override fun onFailure(error: IDJIError) {
                BridgeState.log("RTSP failed: ${error.description()}")
            }
        })
    }

    fun stop() {
        lsm.removeLiveStreamStatusListener(status)
        lsm.stopStream(null)
    }

    fun url(): String =
        "rtsp://${BridgeConfig.RTSP_USER}:${BridgeConfig.RTSP_PASSWORD}@${phoneIp() ?: "<phone-ip>"}:${BridgeConfig.RTSP_PORT}/streaming/live/1"

    companion object {
        /** The phone's IPv4 on Wi-Fi or its own hotspot, which is what the laptop connects to. */
        fun phoneIp(): String? {
            val ifaces = NetworkInterface.getNetworkInterfaces()?.toList() ?: return null
            val preferred = listOf("wlan", "ap", "swlan", "softap")
            return ifaces
                .filter { it.isUp && !it.isLoopback }
                .sortedBy { i -> preferred.indexOfFirst { i.name.startsWith(it) }.let { if (it < 0) 99 else it } }
                .flatMap { it.inetAddresses.toList() }
                .firstOrNull { it is Inet4Address }
                ?.hostAddress
        }
    }
}
