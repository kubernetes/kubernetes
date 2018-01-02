// +build !windows

package sys

import (
	"fmt"
	"os"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/system"
)

// OOMScoreMaxKillable is the maximum score keeping the process killable by the oom killer
const OOMScoreMaxKillable = -999

// SetOOMScore sets the oom score for the provided pid
func SetOOMScore(pid, score int) error {
	path := fmt.Sprintf("/proc/%d/oom_score_adj", pid)
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err = f.WriteString(strconv.Itoa(score)); err != nil {
		if os.IsPermission(err) && system.RunningInUserNS() {
			return nil
		}
		return err
	}
	return nil
}
