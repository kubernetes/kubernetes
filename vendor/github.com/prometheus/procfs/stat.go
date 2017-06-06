package procfs

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Stat represents kernel/system statistics.
type Stat struct {
	// Boot time in seconds since the Epoch.
	BootTime int64
}

// NewStat returns kernel/system statistics read from /proc/stat.
func NewStat() (Stat, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil {
		return Stat{}, err
	}

	return fs.NewStat()
}

// NewStat returns an information about current kernel/system statistics.
func (fs FS) NewStat() (Stat, error) {
	f, err := os.Open(fs.Path("stat"))
	if err != nil {
		return Stat{}, err
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		if !strings.HasPrefix(line, "btime") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 {
			return Stat{}, fmt.Errorf("couldn't parse %s line %s", f.Name(), line)
		}
		i, err := strconv.ParseInt(fields[1], 10, 32)
		if err != nil {
			return Stat{}, fmt.Errorf("couldn't parse %s: %s", fields[1], err)
		}
		return Stat{BootTime: i}, nil
	}
	if err := s.Err(); err != nil {
		return Stat{}, fmt.Errorf("couldn't parse %s: %s", f.Name(), err)
	}

	return Stat{}, fmt.Errorf("couldn't parse %s, missing btime", f.Name())
}
