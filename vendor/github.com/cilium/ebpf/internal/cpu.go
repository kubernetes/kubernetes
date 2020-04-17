package internal

import (
	"fmt"
	"os"
	"sync"

	"github.com/pkg/errors"
)

var sysCPU struct {
	once sync.Once
	err  error
	num  int
}

// PossibleCPUs returns the max number of CPUs a system may possibly have
// Logical CPU numbers must be of the form 0-n
func PossibleCPUs() (int, error) {
	sysCPU.once.Do(func() {
		sysCPU.num, sysCPU.err = parseCPUs("/sys/devices/system/cpu/possible")
	})

	return sysCPU.num, sysCPU.err
}

var onlineCPU struct {
	once sync.Once
	err  error
	num  int
}

// OnlineCPUs returns the number of currently online CPUs
// Logical CPU numbers must be of the form 0-n
func OnlineCPUs() (int, error) {
	onlineCPU.once.Do(func() {
		onlineCPU.num, onlineCPU.err = parseCPUs("/sys/devices/system/cpu/online")
	})

	return onlineCPU.num, onlineCPU.err
}

// parseCPUs parses the number of cpus from sysfs,
// in the format of "/sys/devices/system/cpu/{possible,online,..}.
// Logical CPU numbers must be of the form 0-n
func parseCPUs(path string) (int, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	var low, high int
	n, _ := fmt.Fscanf(file, "%d-%d", &low, &high)
	if n < 1 || low != 0 {
		return 0, errors.Wrapf(err, "%s has unknown format", path)
	}
	if n == 1 {
		high = low
	}

	// cpus is 0 indexed
	return high + 1, nil
}
