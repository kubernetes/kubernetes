package fs2

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
)

func statPSI(dirPath string, file string, stats *cgroups.PSIStats) error {
	if stats == nil {
		return fmt.Errorf("invalid PSIStats pointer is nil")
	}
	f, err := cgroups.OpenFile(dirPath, file, os.O_RDONLY)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		parts := strings.Fields(sc.Text())
		switch parts[0] {
		case "some":
			data, err := parsePSIData(parts[1:])
			if err != nil {
				return err
			}
			stats.Some = data
		case "full":
			data, err := parsePSIData(parts[1:])
			if err != nil {
				return err
			}
			stats.Full = data
		}
	}
	if err := sc.Err(); err != nil {
		return &parseError{Path: dirPath, File: file, Err: err}
	}
	return nil
}

func setFloat(s string, f *float64) error {
	if f == nil {
		return fmt.Errorf("invalid pointer *float64 is nil")
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return fmt.Errorf("invalid PSI value: %q", s)
	}
	*f = v

	return nil
}

func parsePSIData(psi []string) (*cgroups.PSIData, error) {
	data := cgroups.PSIData{
		Avg10:  new(float64),
		Avg60:  new(float64),
		Avg300: new(float64),
		Total:  new(uint64),
	}
	for _, f := range psi {
		kv := strings.SplitN(f, "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid psi data: %q", f)
		}
		switch kv[0] {
		case "avg10":
			if err := setFloat(kv[1], data.Avg10); err != nil {
				return nil, err
			}
		case "avg60":
			if err := setFloat(kv[1], data.Avg60); err != nil {
				return nil, err
			}
		case "avg300":
			if err := setFloat(kv[1], data.Avg300); err != nil {
				return nil, err
			}
		case "total":
			v, err := strconv.ParseUint(kv[1], 10, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid PSI value: %q", f)
			}
			data.Total = &v
		}
	}
	return &data, nil
}
