package fs2

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/cgroups"
)

func statPSI(dirPath string, file string) (*cgroups.PSIStats, error) {
	f, err := cgroups.OpenFile(dirPath, file, os.O_RDONLY)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// Kernel < 4.20, or CONFIG_PSI is not set,
			// or PSI stats are turned off for the cgroup
			// ("echo 0 > cgroup.pressure", kernel >= 6.1).
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var psistats cgroups.PSIStats
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		parts := strings.Fields(sc.Text())
		var pv *cgroups.PSIData
		switch parts[0] {
		case "some":
			pv = &psistats.Some
		case "full":
			pv = &psistats.Full
		}
		if pv != nil {
			*pv, err = parsePSIData(parts[1:])
			if err != nil {
				return nil, &parseError{Path: dirPath, File: file, Err: err}
			}
		}
	}
	if err := sc.Err(); err != nil {
		if errors.Is(err, unix.ENOTSUP) {
			// Some kernels (e.g. CS9) may return ENOTSUP on read
			// if psi=1 kernel cmdline parameter is required.
			return nil, nil
		}
		return nil, &parseError{Path: dirPath, File: file, Err: err}
	}
	return &psistats, nil
}

func parsePSIData(psi []string) (cgroups.PSIData, error) {
	data := cgroups.PSIData{}
	for _, f := range psi {
		kv := strings.SplitN(f, "=", 2)
		if len(kv) != 2 {
			return data, fmt.Errorf("invalid psi data: %q", f)
		}
		var pv *float64
		switch kv[0] {
		case "avg10":
			pv = &data.Avg10
		case "avg60":
			pv = &data.Avg60
		case "avg300":
			pv = &data.Avg300
		case "total":
			v, err := strconv.ParseUint(kv[1], 10, 64)
			if err != nil {
				return data, fmt.Errorf("invalid %s PSI value: %w", kv[0], err)
			}
			data.Total = v
		}
		if pv != nil {
			v, err := strconv.ParseFloat(kv[1], 64)
			if err != nil {
				return data, fmt.Errorf("invalid %s PSI value: %w", kv[0], err)
			}
			*pv = v
		}
	}
	return data, nil
}
