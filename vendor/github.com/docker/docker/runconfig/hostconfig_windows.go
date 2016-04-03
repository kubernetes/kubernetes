package runconfig

func (n NetworkMode) IsDefault() bool {
	return n == "default"
}

func DefaultDaemonNetworkMode() NetworkMode {
	return NetworkMode("default")
}

func (n NetworkMode) NetworkName() string {
	if n.IsDefault() {
		return "default"
	}
	return ""
}
