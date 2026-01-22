package fscommon

import (
	"bufio"
	"errors"
	"math"
	"os"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/opencontainers/cgroups"
)

// parseRdmaKV parses raw string to RdmaEntry.
func parseRdmaKV(raw string, entry *cgroups.RdmaEntry) error {
	var value uint32

	k, v, ok := strings.Cut(raw, "=")

	if !ok {
		return errors.New("Unable to parse RDMA entry")
	}

	if v == "max" {
		value = math.MaxUint32
	} else {
		val64, err := strconv.ParseUint(v, 10, 32)
		if err != nil {
			return err
		}
		value = uint32(val64)
	}
	switch k {
	case "hca_handle":
		entry.HcaHandles = value
	case "hca_object":
		entry.HcaObjects = value
	}

	return nil
}

// readRdmaEntries reads and converts array of rawstrings to RdmaEntries from file.
// example entry: mlx4_0 hca_handle=2 hca_object=2000
func readRdmaEntries(dir, file string) ([]cgroups.RdmaEntry, error) {
	rdmaEntries := make([]cgroups.RdmaEntry, 0)
	fd, err := cgroups.OpenFile(dir, file, unix.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	scanner := bufio.NewScanner(fd)
	for scanner.Scan() {
		parts := strings.SplitN(scanner.Text(), " ", 4)
		if len(parts) == 3 {
			entry := new(cgroups.RdmaEntry)
			entry.Device = parts[0]
			err = parseRdmaKV(parts[1], entry)
			if err != nil {
				continue
			}
			err = parseRdmaKV(parts[2], entry)
			if err != nil {
				continue
			}

			rdmaEntries = append(rdmaEntries, *entry)
		}
	}
	return rdmaEntries, scanner.Err()
}

// RdmaGetStats returns rdma stats such as totalLimit and current entries.
func RdmaGetStats(path string, stats *cgroups.Stats) error {
	currentEntries, err := readRdmaEntries(path, "rdma.current")
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			err = nil
		}
		return err
	}
	maxEntries, err := readRdmaEntries(path, "rdma.max")
	if err != nil {
		return err
	}
	// If device got removed between reading two files, ignore returning stats.
	if len(currentEntries) != len(maxEntries) {
		return nil
	}

	stats.RdmaStats = cgroups.RdmaStats{
		RdmaLimit:   maxEntries,
		RdmaCurrent: currentEntries,
	}

	return nil
}

func createCmdString(device string, limits cgroups.LinuxRdma) string {
	cmdString := device
	if limits.HcaHandles != nil {
		cmdString += " hca_handle=" + strconv.FormatUint(uint64(*limits.HcaHandles), 10)
	}
	if limits.HcaObjects != nil {
		cmdString += " hca_object=" + strconv.FormatUint(uint64(*limits.HcaObjects), 10)
	}
	return cmdString
}

// RdmaSet sets RDMA resources.
func RdmaSet(path string, r *cgroups.Resources) error {
	for device, limits := range r.Rdma {
		if err := cgroups.WriteFile(path, "rdma.max", createCmdString(device, limits)); err != nil {
			return err
		}
	}
	return nil
}
