package fs2

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"

	"github.com/opencontainers/cgroups"
)

func isIoSet(r *cgroups.Resources) bool {
	return r.BlkioWeight != 0 ||
		len(r.BlkioWeightDevice) > 0 ||
		len(r.BlkioThrottleReadBpsDevice) > 0 ||
		len(r.BlkioThrottleWriteBpsDevice) > 0 ||
		len(r.BlkioThrottleReadIOPSDevice) > 0 ||
		len(r.BlkioThrottleWriteIOPSDevice) > 0
}

// bfqDeviceWeightSupported checks for per-device BFQ weight support (added
// in kernel v5.4, commit 795fe54c2a8) by reading from "io.bfq.weight".
func bfqDeviceWeightSupported(bfq *os.File) bool {
	if bfq == nil {
		return false
	}
	_, _ = bfq.Seek(0, 0)
	buf := make([]byte, 32)
	_, _ = bfq.Read(buf)
	// If only a single number (default weight) if read back, we have older kernel.
	_, err := strconv.ParseInt(string(bytes.TrimSpace(buf)), 10, 64)
	return err != nil
}

func setIo(dirPath string, r *cgroups.Resources) error {
	if !isIoSet(r) {
		return nil
	}

	// If BFQ IO scheduler is available, use it.
	var bfq *os.File
	if r.BlkioWeight != 0 || len(r.BlkioWeightDevice) > 0 {
		var err error
		bfq, err = cgroups.OpenFile(dirPath, "io.bfq.weight", os.O_RDWR)
		if err == nil {
			defer bfq.Close()
		} else if !os.IsNotExist(err) {
			return err
		}
	}

	if r.BlkioWeight != 0 {
		if bfq != nil { // Use BFQ.
			if _, err := bfq.WriteString(strconv.FormatUint(uint64(r.BlkioWeight), 10)); err != nil {
				return err
			}
		} else {
			// Fallback to io.weight with a conversion scheme.
			v := cgroups.ConvertBlkIOToIOWeightValue(r.BlkioWeight)
			if err := cgroups.WriteFile(dirPath, "io.weight", strconv.FormatUint(v, 10)); err != nil {
				return err
			}
		}
	}
	if bfqDeviceWeightSupported(bfq) {
		for _, wd := range r.BlkioWeightDevice {
			if _, err := bfq.WriteString(wd.WeightString() + "\n"); err != nil {
				return fmt.Errorf("setting device weight %q: %w", wd.WeightString(), err)
			}
		}
	}
	for _, td := range r.BlkioThrottleReadBpsDevice {
		if err := cgroups.WriteFile(dirPath, "io.max", td.StringName("rbps")); err != nil {
			return err
		}
	}
	for _, td := range r.BlkioThrottleWriteBpsDevice {
		if err := cgroups.WriteFile(dirPath, "io.max", td.StringName("wbps")); err != nil {
			return err
		}
	}
	for _, td := range r.BlkioThrottleReadIOPSDevice {
		if err := cgroups.WriteFile(dirPath, "io.max", td.StringName("riops")); err != nil {
			return err
		}
	}
	for _, td := range r.BlkioThrottleWriteIOPSDevice {
		if err := cgroups.WriteFile(dirPath, "io.max", td.StringName("wiops")); err != nil {
			return err
		}
	}

	return nil
}

func readCgroup2MapFile(dirPath string, name string) (map[string][]string, error) {
	ret := map[string][]string{}
	f, err := cgroups.OpenFile(dirPath, name, os.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		ret[parts[0]] = parts[1:]
	}
	if err := scanner.Err(); err != nil {
		return nil, &parseError{Path: dirPath, File: name, Err: err}
	}
	return ret, nil
}

func statIo(dirPath string, stats *cgroups.Stats) error {
	const file = "io.stat"
	values, err := readCgroup2MapFile(dirPath, file)
	if err != nil {
		return err
	}
	// more details on the io.stat file format: https://www.kernel.org/doc/Documentation/cgroup-v2.txt
	var parsedStats cgroups.BlkioStats
	for k, v := range values {
		d := strings.Split(k, ":")
		if len(d) != 2 {
			continue
		}
		major, err := strconv.ParseUint(d[0], 10, 64)
		if err != nil {
			return &parseError{Path: dirPath, File: file, Err: err}
		}
		minor, err := strconv.ParseUint(d[1], 10, 64)
		if err != nil {
			return &parseError{Path: dirPath, File: file, Err: err}
		}

		for _, item := range v {
			d := strings.Split(item, "=")
			if len(d) != 2 {
				continue
			}
			op := d[0]

			// Map to the cgroupv1 naming and layout (in separate tables).
			var targetTable *[]cgroups.BlkioStatEntry
			switch op {
			// Equivalent to cgroupv1's blkio.io_service_bytes.
			case "rbytes":
				op = "Read"
				targetTable = &parsedStats.IoServiceBytesRecursive
			case "wbytes":
				op = "Write"
				targetTable = &parsedStats.IoServiceBytesRecursive
			// Equivalent to cgroupv1's blkio.io_serviced.
			case "rios":
				op = "Read"
				targetTable = &parsedStats.IoServicedRecursive
			case "wios":
				op = "Write"
				targetTable = &parsedStats.IoServicedRecursive

			case "cost.usage":
				op = "Count"
				targetTable = &parsedStats.IoCostUsage
			case "cost.wait":
				op = "Count"
				targetTable = &parsedStats.IoCostWait
			case "cost.indebt":
				op = "Count"
				targetTable = &parsedStats.IoCostIndebt
			case "cost.indelay":
				op = "Count"
				targetTable = &parsedStats.IoCostIndelay

			default:
				logrus.Debugf("cgroupv2 io stats: unknown entry %s", item)
				continue
			}

			value, err := strconv.ParseUint(d[1], 10, 64)
			if err != nil {
				return &parseError{Path: dirPath, File: file, Err: err}
			}

			entry := cgroups.BlkioStatEntry{
				Op:    op,
				Major: major,
				Minor: minor,
				Value: value,
			}
			*targetTable = append(*targetTable, entry)
		}
	}
	stats.BlkioStats = parsedStats
	return nil
}
