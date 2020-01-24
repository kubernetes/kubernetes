// +build linux

package fs2

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

func setIo(dirPath string, cgroup *configs.Cgroup) error {
	if cgroup.Resources.BlkioWeight != 0 {
		filename := "io.bfq.weight"
		if err := fscommon.WriteFile(dirPath, filename, strconv.FormatUint(uint64(cgroup.Resources.BlkioWeight), 10)); err != nil {
			return err
		}
	}

	for _, td := range cgroup.Resources.BlkioThrottleReadBpsDevice {
		if err := fscommon.WriteFile(dirPath, "io.max", td.StringName("rbps")); err != nil {
			return err
		}
	}
	for _, td := range cgroup.Resources.BlkioThrottleWriteBpsDevice {
		if err := fscommon.WriteFile(dirPath, "io.max", td.StringName("wbps")); err != nil {
			return err
		}
	}
	for _, td := range cgroup.Resources.BlkioThrottleReadIOPSDevice {
		if err := fscommon.WriteFile(dirPath, "io.max", td.StringName("riops")); err != nil {
			return err
		}
	}
	for _, td := range cgroup.Resources.BlkioThrottleWriteIOPSDevice {
		if err := fscommon.WriteFile(dirPath, "io.max", td.StringName("wiops")); err != nil {
			return err
		}
	}

	return nil
}

func readCgroup2MapFile(dirPath string, name string) (map[string][]string, error) {
	ret := map[string][]string{}
	p := filepath.Join(dirPath, name)
	f, err := os.Open(p)
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
		return nil, err
	}
	return ret, nil
}

func statIo(dirPath string, stats *cgroups.Stats) error {
	// more details on the io.stat file format: https://www.kernel.org/doc/Documentation/cgroup-v2.txt
	var ioServiceBytesRecursive []cgroups.BlkioStatEntry
	values, err := readCgroup2MapFile(dirPath, "io.stat")
	if err != nil {
		return err
	}
	for k, v := range values {
		d := strings.Split(k, ":")
		if len(d) != 2 {
			continue
		}
		minor, err := strconv.ParseUint(d[0], 10, 0)
		if err != nil {
			return err
		}
		major, err := strconv.ParseUint(d[1], 10, 0)
		if err != nil {
			return err
		}

		for _, item := range v {
			d := strings.Split(item, "=")
			if len(d) != 2 {
				continue
			}
			op := d[0]

			// Accommodate the cgroup v1 naming
			switch op {
			case "rbytes":
				op = "read"
			case "wbytes":
				op = "write"
			}

			value, err := strconv.ParseUint(d[1], 10, 0)
			if err != nil {
				return err
			}

			entry := cgroups.BlkioStatEntry{
				Op:    op,
				Major: major,
				Minor: minor,
				Value: value,
			}
			ioServiceBytesRecursive = append(ioServiceBytesRecursive, entry)
		}
	}
	stats.BlkioStats = cgroups.BlkioStats{IoServiceBytesRecursive: ioServiceBytesRecursive}
	return nil
}
