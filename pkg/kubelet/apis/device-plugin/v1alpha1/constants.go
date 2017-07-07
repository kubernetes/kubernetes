package deviceplugin

const (
	Healty    = "Healthy"
	Unhealthy = "Unhealthy"

	HeartbeatOk    = "Heartbeat ok"
	HeartbeatKo    = "Heartbeat ko"
	HeartbeatError = "Heartbeat Error"

	Version          = "0.1"
	DevicePluginPath = "/var/run/kubernetes/"
	KubeletSocket    = DevicePluginPath + "kubelet.sock"
)
