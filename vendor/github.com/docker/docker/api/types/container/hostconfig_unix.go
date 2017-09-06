// +build !windows

package container

// IsValid indicates if an isolation technology is valid
func (i Isolation) IsValid() bool {
	return i.IsDefault()
}

// NetworkName returns the name of the network stack.
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
	} else if n.IsUserDefined() {
		return n.UserDefined()
	}
	return ""
}

// IsBridge indicates whether container uses the bridge network stack
func (n NetworkMode) IsBridge() bool {
	return n == "bridge"
}

// IsHost indicates whether container uses the host network stack.
func (n NetworkMode) IsHost() bool {
	return n == "host"
}

// IsUserDefined indicates user-created network
func (n NetworkMode) IsUserDefined() bool {
	return !n.IsDefault() && !n.IsBridge() && !n.IsHost() && !n.IsNone() && !n.IsContainer()
}
