package proxyproto

// ProtocolVersionAndCommand represents the command in proxy protocol v2.
// Command doesn't exist in v1 but it should be set since other parts of
// this library may rely on it for determining connection details.
type ProtocolVersionAndCommand byte

const (
	// LOCAL represents the LOCAL command in v2 or UNKNOWN transport in v1,
	// in which case no address information is expected.
	LOCAL ProtocolVersionAndCommand = '\x20'
	// PROXY represents the PROXY command in v2 or transport is not UNKNOWN in v1,
	// in which case valid local/remote address and port information is expected.
	PROXY ProtocolVersionAndCommand = '\x21'
)

var supportedCommand = map[ProtocolVersionAndCommand]bool{
	LOCAL: true,
	PROXY: true,
}

// IsLocal returns true if the command in v2 is LOCAL or the transport in v1 is UNKNOWN,
// i.e. when no address information is expected, false otherwise.
func (pvc ProtocolVersionAndCommand) IsLocal() bool {
	return LOCAL == pvc
}

// IsProxy returns true if the command in v2 is PROXY or the transport in v1 is not UNKNOWN,
// i.e. when valid local/remote address and port information is expected, false otherwise.
func (pvc ProtocolVersionAndCommand) IsProxy() bool {
	return PROXY == pvc
}

// IsUnspec returns true if the command is unspecified, false otherwise.
func (pvc ProtocolVersionAndCommand) IsUnspec() bool {
	return !(pvc.IsLocal() || pvc.IsProxy())
}

func (pvc ProtocolVersionAndCommand) toByte() byte {
	if pvc.IsLocal() {
		return byte(LOCAL)
	} else if pvc.IsProxy() {
		return byte(PROXY)
	}

	return byte(LOCAL)
}
