package netlink

// According to linux/include/uapi/linux/unix_diag.h
const (
	UNIX_DIAG_NAME = iota
	UNIX_DIAG_VFS
	UNIX_DIAG_PEER
	UNIX_DIAG_ICONS
	UNIX_DIAG_RQLEN
	UNIX_DIAG_MEMINFO
	UNIX_DIAG_SHUTDOWN
	UNIX_DIAG_UID
	UNIX_DIAG_MAX
)

type UnixDiagInfoResp struct {
	DiagMsg  *UnixSocket
	Name     *string
	Peer     *uint32
	Queue    *QueueInfo
	Shutdown *uint8
}

type QueueInfo struct {
	RQueue uint32
	WQueue uint32
}
