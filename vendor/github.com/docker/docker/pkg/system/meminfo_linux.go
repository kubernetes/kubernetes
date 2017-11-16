package system

import (
	"bufio"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/docker/go-units"
)

// ReadMemInfo retrieves memory statistics of the host system and returns a
// MemInfo type.
func ReadMemInfo() (*MemInfo, error) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return parseMemInfo(file)
}

// parseMemInfo parses the /proc/meminfo file into
// a MemInfo object given an io.Reader to the file.
// Throws error if there are problems reading from the file
func parseMemInfo(reader io.Reader) (*MemInfo, error) {
	meminfo := &MemInfo{}
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		// Expected format: ["MemTotal:", "1234", "kB"]
		parts := strings.Fields(scanner.Text())

		// Sanity checks: Skip malformed entries.
		if len(parts) < 3 || parts[2] != "kB" {
			continue
		}

		// Convert to bytes.
		size, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		bytes := int64(size) * units.KiB

		switch parts[0] {
		case "MemTotal:":
			meminfo.MemTotal = bytes
		case "MemFree:":
			meminfo.MemFree = bytes
		case "SwapTotal:":
			meminfo.SwapTotal = bytes
		case "SwapFree:":
			meminfo.SwapFree = bytes
		}

	}

	// Handle errors that may have occurred during the reading of the file.
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return meminfo, nil
}
