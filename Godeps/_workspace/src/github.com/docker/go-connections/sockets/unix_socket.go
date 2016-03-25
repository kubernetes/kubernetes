// +build linux freebsd solaris

package sockets

import (
	"fmt"
	"net"
	"os"
	"strconv"
	"syscall"

	"github.com/Sirupsen/logrus"
	"github.com/opencontainers/runc/libcontainer/user"
)

// NewUnixSocket creates a unix socket with the specified path and group.
func NewUnixSocket(path, group string) (net.Listener, error) {
	if err := syscall.Unlink(path); err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	mask := syscall.Umask(0777)
	defer syscall.Umask(mask)
	l, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}
	if err := setSocketGroup(path, group); err != nil {
		l.Close()
		return nil, err
	}
	if err := os.Chmod(path, 0660); err != nil {
		l.Close()
		return nil, err
	}
	return l, nil
}

func setSocketGroup(path, group string) error {
	if group == "" {
		return nil
	}
	if err := changeGroup(path, group); err != nil {
		if group != "docker" {
			return err
		}
		logrus.Debugf("Warning: could not change group %s to docker: %v", path, err)
	}
	return nil
}

func changeGroup(path string, nameOrGid string) error {
	gid, err := lookupGidByName(nameOrGid)
	if err != nil {
		return err
	}
	logrus.Debugf("%s group found. gid: %d", nameOrGid, gid)
	return os.Chown(path, 0, gid)
}

func lookupGidByName(nameOrGid string) (int, error) {
	groupFile, err := user.GetGroupPath()
	if err != nil {
		return -1, err
	}
	groups, err := user.ParseGroupFileFilter(groupFile, func(g user.Group) bool {
		return g.Name == nameOrGid || strconv.Itoa(g.Gid) == nameOrGid
	})
	if err != nil {
		return -1, err
	}
	if groups != nil && len(groups) > 0 {
		return groups[0].Gid, nil
	}
	gid, err := strconv.Atoi(nameOrGid)
	if err == nil {
		logrus.Warnf("Could not find GID %d", gid)
		return gid, nil
	}
	return -1, fmt.Errorf("Group %s not found", nameOrGid)
}
