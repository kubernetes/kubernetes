// +build linux,nokmem

package fs

import (
	"errors"
)

func EnableKernelMemoryAccounting(path string) error {
	return nil
}

func setKernelMemory(path string, kernelMemoryLimit int64) error {
	return errors.New("kernel memory accounting disabled in this runc build")
}
