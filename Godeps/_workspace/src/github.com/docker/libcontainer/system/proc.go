package system

import (
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
)

// look in /proc to find the process start time so that we can verify
// that this pid has started after ourself
func GetProcessStartTime(pid int) (string, error) {
	data, err := ioutil.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "stat"))
	if err != nil {
		return "", err
	}

	parts := strings.Split(string(data), " ")
	// the starttime is located at pos 22
	// from the man page
	//
	// starttime %llu (was %lu before Linux 2.6)
	// (22)  The  time the process started after system boot.  In kernels before Linux 2.6, this
	// value was expressed in jiffies.  Since Linux 2.6, the value is expressed in  clock  ticks
	// (divide by sysconf(_SC_CLK_TCK)).
	return parts[22-1], nil // starts at 1
}
