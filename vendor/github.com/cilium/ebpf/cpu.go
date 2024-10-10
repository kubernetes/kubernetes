package ebpf

import (
	"fmt"
	"os"
	"strings"
	"sync"
)

var possibleCPU = sync.OnceValues(func() (int, error) {
	return parseCPUsFromFile("/sys/devices/system/cpu/possible")
})

// PossibleCPU returns the max number of CPUs a system may possibly have
// Logical CPU numbers must be of the form 0-n
func PossibleCPU() (int, error) {
	return possibleCPU()
}

// MustPossibleCPU is a helper that wraps a call to PossibleCPU and panics if
// the error is non-nil.
func MustPossibleCPU() int {
	cpus, err := PossibleCPU()
	if err != nil {
		panic(err)
	}
	return cpus
}

func parseCPUsFromFile(path string) (int, error) {
	spec, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}

	n, err := parseCPUs(string(spec))
	if err != nil {
		return 0, fmt.Errorf("can't parse %s: %v", path, err)
	}

	return n, nil
}

// parseCPUs parses the number of cpus from a string produced
// by bitmap_list_string() in the Linux kernel.
// Multiple ranges are rejected, since they can't be unified
// into a single number.
// This is the format of /sys/devices/system/cpu/possible, it
// is not suitable for /sys/devices/system/cpu/online, etc.
func parseCPUs(spec string) (int, error) {
	if strings.Trim(spec, "\n") == "0" {
		return 1, nil
	}

	var low, high int
	n, err := fmt.Sscanf(spec, "%d-%d\n", &low, &high)
	if n != 2 || err != nil {
		return 0, fmt.Errorf("invalid format: %s", spec)
	}
	if low != 0 {
		return 0, fmt.Errorf("CPU spec doesn't start at zero: %s", spec)
	}

	// cpus is 0 indexed
	return high + 1, nil
}
