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
	return parseStartTime(string(data))
}

func parseStartTime(stat string) (string, error) {
	// the starttime is located at pos 22
	// from the man page
	//
	// starttime %llu (was %lu before Linux 2.6)
	// (22)  The  time the process started after system boot.  In kernels before Linux 2.6, this
	// value was expressed in jiffies.  Since Linux 2.6, the value is expressed in  clock  ticks
	// (divide by sysconf(_SC_CLK_TCK)).
	//
	// NOTE:
	// pos 2 could contain space and is inside `(` and `)`:
	// (2) comm  %s
	// The filename of the executable, in parentheses.
	// This is visible whether or not the executable is
	// swapped out.
	//
	// the following is an example:
	// 89653 (gunicorn: maste) S 89630 89653 89653 0 -1 4194560 29689 28896 0 3 146 32 76 19 20 0 1 0 2971844 52965376 3920 18446744073709551615 1 1 0 0 0 0 0 16781312 137447943 0 0 0 17 1 0 0 0 0 0 0 0 0 0 0 0 0 0

	// get parts after last `)`:
	s := strings.Split(stat, ")")
	parts := strings.Split(strings.TrimSpace(s[len(s)-1]), " ")
	return parts[22-3], nil // starts at 3 (after the filename pos `2`)
}
