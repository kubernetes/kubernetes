// +build linux freebsd

package libcontainer

// cgroup restoring strategy provided by criu
type cg_mode uint32

const (
	CRIU_CG_MODE_SOFT    cg_mode = 3 + iota // restore cgroup properties if only dir created by criu
	CRIU_CG_MODE_FULL                       // always restore all cgroups and their properties
	CRIU_CG_MODE_STRICT                     // restore all, requiring them to not present in the system
	CRIU_CG_MODE_DEFAULT                    // the same as CRIU_CG_MODE_SOFT
)

type CriuPageServerInfo struct {
	Address string // IP address of CRIU page server
	Port    int32  // port number of CRIU page server
}

type VethPairName struct {
	ContainerInterfaceName string
	HostInterfaceName      string
}

type CriuOpts struct {
	ImagesDirectory         string             // directory for storing image files
	WorkDirectory           string             // directory to cd and write logs/pidfiles/stats to
	LeaveRunning            bool               // leave container in running state after checkpoint
	TcpEstablished          bool               // checkpoint/restore established TCP connections
	ExternalUnixConnections bool               // allow external unix connections
	ShellJob                bool               // allow to dump and restore shell jobs
	FileLocks               bool               // handle file locks, for safety
	PageServer              CriuPageServerInfo // allow to dump to criu page server
	VethPairs               []VethPairName     // pass the veth to criu when restore
	ManageCgroupsMode       cg_mode            // dump or restore cgroup mode
}
