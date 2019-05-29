// +build linux,nokmem

package fs

func EnableKernelMemoryAccounting(path string) error {
	return nil
}

func setKernelMemory(path string, kernelMemoryLimit int64) error {
	return nil
}
