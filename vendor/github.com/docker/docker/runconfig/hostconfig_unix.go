// +build !windows

package runconfig

import (
	"strings"
)

// IsPrivate indicates whether container use it's private network stack
func (n NetworkMode) IsPrivate() bool {
	return !(n.IsHost() || n.IsContainer())
}

func (n NetworkMode) IsDefault() bool {
	return n == "default"
}

func DefaultDaemonNetworkMode() NetworkMode {
	return NetworkMode("bridge")
}

func (n NetworkMode) NetworkName() string {
	if n.IsBridge() {
		return "bridge"
	} else if n.IsHost() {
		return "host"
	} else if n.IsContainer() {
		return "container"
	} else if n.IsNone() {
		return "none"
	} else if n.IsDefault() {
		return "default"
	}
	return ""
}

func (n NetworkMode) IsBridge() bool {
	return n == "bridge"
}

func (n NetworkMode) IsHost() bool {
	return n == "host"
}

func (n NetworkMode) IsContainer() bool {
	parts := strings.SplitN(string(n), ":", 2)
	return len(parts) > 1 && parts[0] == "container"
}

func (n NetworkMode) IsNone() bool {
	return n == "none"
}
