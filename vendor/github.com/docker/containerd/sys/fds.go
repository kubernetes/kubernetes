// +build !windows,!darwin

package sys

import (
	"io/ioutil"
	"path/filepath"
	"strconv"
)

// GetOpenFds returns the number of open fds for the process provided by pid
func GetOpenFds(pid int) (int, error) {
	dirs, err := ioutil.ReadDir(filepath.Join("/proc", strconv.Itoa(pid), "fd"))
	if err != nil {
		return -1, err
	}
	return len(dirs), nil
}
