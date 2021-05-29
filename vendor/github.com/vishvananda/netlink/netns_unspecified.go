// +build !linux

package netlink

func GetNetNsIdByPid(pid int) (int, error) {
	return 0, ErrNotImplemented
}

func SetNetNsIdByPid(pid, nsid int) error {
	return ErrNotImplemented
}

func GetNetNsIdByFd(fd int) (int, error) {
	return 0, ErrNotImplemented
}

func SetNetNsIdByFd(fd, nsid int) error {
	return ErrNotImplemented
}
