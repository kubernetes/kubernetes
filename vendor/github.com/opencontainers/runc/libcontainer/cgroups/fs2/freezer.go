package fs2

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

func setFreezer(dirPath string, state configs.FreezerState) error {
	var stateStr string
	switch state {
	case configs.Undefined:
		return nil
	case configs.Frozen:
		stateStr = "1"
	case configs.Thawed:
		stateStr = "0"
	default:
		return fmt.Errorf("invalid freezer state %q requested", state)
	}

	fd, err := cgroups.OpenFile(dirPath, "cgroup.freeze", unix.O_RDWR)
	if err != nil {
		// We can ignore this request as long as the user didn't ask us to
		// freeze the container (since without the freezer cgroup, that's a
		// no-op).
		if state != configs.Frozen {
			return nil
		}
		return fmt.Errorf("freezer not supported: %w", err)
	}
	defer fd.Close()

	if _, err := fd.WriteString(stateStr); err != nil {
		return err
	}
	// Confirm that the cgroup did actually change states.
	if actualState, err := readFreezer(dirPath, fd); err != nil {
		return err
	} else if actualState != state {
		return fmt.Errorf(`expected "cgroup.freeze" to be in state %q but was in %q`, state, actualState)
	}
	return nil
}

func getFreezer(dirPath string) (configs.FreezerState, error) {
	fd, err := cgroups.OpenFile(dirPath, "cgroup.freeze", unix.O_RDONLY)
	if err != nil {
		// If the kernel is too old, then we just treat the freezer as being in
		// an "undefined" state.
		if os.IsNotExist(err) || errors.Is(err, unix.ENODEV) {
			err = nil
		}
		return configs.Undefined, err
	}
	defer fd.Close()

	return readFreezer(dirPath, fd)
}

func readFreezer(dirPath string, fd *os.File) (configs.FreezerState, error) {
	if _, err := fd.Seek(0, 0); err != nil {
		return configs.Undefined, err
	}
	state := make([]byte, 2)
	if _, err := fd.Read(state); err != nil {
		return configs.Undefined, err
	}
	switch string(state) {
	case "0\n":
		return configs.Thawed, nil
	case "1\n":
		return waitFrozen(dirPath)
	default:
		return configs.Undefined, fmt.Errorf(`unknown "cgroup.freeze" state: %q`, state)
	}
}

// waitFrozen polls cgroup.events until it sees "frozen 1" in it.
func waitFrozen(dirPath string) (configs.FreezerState, error) {
	fd, err := cgroups.OpenFile(dirPath, "cgroup.events", unix.O_RDONLY)
	if err != nil {
		return configs.Undefined, err
	}
	defer fd.Close()

	// XXX: Simple wait/read/retry is used here. An implementation
	// based on poll(2) or inotify(7) is possible, but it makes the code
	// much more complicated. Maybe address this later.
	const (
		// Perform maxIter with waitTime in between iterations.
		waitTime = 10 * time.Millisecond
		maxIter  = 1000
	)
	scanner := bufio.NewScanner(fd)
	for i := 0; scanner.Scan(); {
		if i == maxIter {
			return configs.Undefined, fmt.Errorf("timeout of %s reached waiting for the cgroup to freeze", waitTime*maxIter)
		}
		line := scanner.Text()
		val := strings.TrimPrefix(line, "frozen ")
		if val != line { // got prefix
			if val[0] == '1' {
				return configs.Frozen, nil
			}

			i++
			// wait, then re-read
			time.Sleep(waitTime)
			_, err := fd.Seek(0, 0)
			if err != nil {
				return configs.Undefined, err
			}
		}
	}
	// Should only reach here either on read error,
	// or if the file does not contain "frozen " line.
	return configs.Undefined, scanner.Err()
}
