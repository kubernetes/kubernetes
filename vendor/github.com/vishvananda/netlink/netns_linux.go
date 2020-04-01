package netlink

// Network namespace ID functions
//
// The kernel has a weird concept called the network namespace ID.
// This is different from the file reference in proc (and any bind-mounted
// namespaces, etc.)
//
// Instead, namespaces can be assigned a numeric ID at any time. Once set,
// the ID is fixed. The ID can either be set manually by the user, or
// automatically, triggered by certain kernel actions. The most common kernel
// action that triggers namespace ID creation is moving one end of a veth pair
// in to that namespace.

import (
	"fmt"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

// These can be replaced by the values from sys/unix when it is next released.
const (
	_ = iota
	NETNSA_NSID
	NETNSA_PID
	NETNSA_FD
)

// GetNetNsIdByPid looks up the network namespace ID for a given pid (really thread id).
// Returns -1 if the namespace does not have an ID set.
func (h *Handle) GetNetNsIdByPid(pid int) (int, error) {
	return h.getNetNsId(NETNSA_PID, uint32(pid))
}

// GetNetNsIdByPid looks up the network namespace ID for a given pid (really thread id).
// Returns -1 if the namespace does not have an ID set.
func GetNetNsIdByPid(pid int) (int, error) {
	return pkgHandle.GetNetNsIdByPid(pid)
}

// SetNetNSIdByPid sets the ID of the network namespace for a given pid (really thread id).
// The ID can only be set for namespaces without an ID already set.
func (h *Handle) SetNetNsIdByPid(pid, nsid int) error {
	return h.setNetNsId(NETNSA_PID, uint32(pid), uint32(nsid))
}

// SetNetNSIdByPid sets the ID of the network namespace for a given pid (really thread id).
// The ID can only be set for namespaces without an ID already set.
func SetNetNsIdByPid(pid, nsid int) error {
	return pkgHandle.SetNetNsIdByPid(pid, nsid)
}

// GetNetNsIdByFd looks up the network namespace ID for a given fd.
// fd must be an open file descriptor to a namespace file.
// Returns -1 if the namespace does not have an ID set.
func (h *Handle) GetNetNsIdByFd(fd int) (int, error) {
	return h.getNetNsId(NETNSA_FD, uint32(fd))
}

// GetNetNsIdByFd looks up the network namespace ID for a given fd.
// fd must be an open file descriptor to a namespace file.
// Returns -1 if the namespace does not have an ID set.
func GetNetNsIdByFd(fd int) (int, error) {
	return pkgHandle.GetNetNsIdByFd(fd)
}

// SetNetNSIdByFd sets the ID of the network namespace for a given fd.
// fd must be an open file descriptor to a namespace file.
// The ID can only be set for namespaces without an ID already set.
func (h *Handle) SetNetNsIdByFd(fd, nsid int) error {
	return h.setNetNsId(NETNSA_FD, uint32(fd), uint32(nsid))
}

// SetNetNSIdByFd sets the ID of the network namespace for a given fd.
// fd must be an open file descriptor to a namespace file.
// The ID can only be set for namespaces without an ID already set.
func SetNetNsIdByFd(fd, nsid int) error {
	return pkgHandle.SetNetNsIdByFd(fd, nsid)
}

// getNetNsId requests the netnsid for a given type-val pair
// type should be either NETNSA_PID or NETNSA_FD
func (h *Handle) getNetNsId(attrType int, val uint32) (int, error) {
	req := h.newNetlinkRequest(unix.RTM_GETNSID, unix.NLM_F_REQUEST)

	rtgen := nl.NewRtGenMsg()
	req.AddData(rtgen)

	b := make([]byte, 4, 4)
	native.PutUint32(b, val)
	attr := nl.NewRtAttr(attrType, b)
	req.AddData(attr)

	msgs, err := req.Execute(unix.NETLINK_ROUTE, unix.RTM_NEWNSID)

	if err != nil {
		return 0, err
	}

	for _, m := range msgs {
		msg := nl.DeserializeRtGenMsg(m)

		attrs, err := nl.ParseRouteAttr(m[msg.Len():])
		if err != nil {
			return 0, err
		}

		for _, attr := range attrs {
			switch attr.Attr.Type {
			case NETNSA_NSID:
				return int(int32(native.Uint32(attr.Value))), nil
			}
		}
	}

	return 0, fmt.Errorf("unexpected empty result")
}

// setNetNsId sets the netnsid for a given type-val pair
// type should be either NETNSA_PID or NETNSA_FD
// The ID can only be set for namespaces without an ID already set
func (h *Handle) setNetNsId(attrType int, val uint32, newnsid uint32) error {
	req := h.newNetlinkRequest(unix.RTM_NEWNSID, unix.NLM_F_REQUEST|unix.NLM_F_ACK)

	rtgen := nl.NewRtGenMsg()
	req.AddData(rtgen)

	b := make([]byte, 4, 4)
	native.PutUint32(b, val)
	attr := nl.NewRtAttr(attrType, b)
	req.AddData(attr)

	b1 := make([]byte, 4, 4)
	native.PutUint32(b1, newnsid)
	attr1 := nl.NewRtAttr(NETNSA_NSID, b1)
	req.AddData(attr1)

	_, err := req.Execute(unix.NETLINK_ROUTE, unix.RTM_NEWNSID)
	return err
}
