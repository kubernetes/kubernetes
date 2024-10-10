package kallsyms

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"sync"
)

var kernelModules struct {
	sync.RWMutex
	// function to kernel module mapping
	kmods map[string]string
}

// KernelModule returns the kernel module, if any, a probe-able function is contained in.
func KernelModule(fn string) (string, error) {
	kernelModules.RLock()
	kmods := kernelModules.kmods
	kernelModules.RUnlock()

	if kmods == nil {
		kernelModules.Lock()
		defer kernelModules.Unlock()
		kmods = kernelModules.kmods
	}

	if kmods != nil {
		return kmods[fn], nil
	}

	f, err := os.Open("/proc/kallsyms")
	if err != nil {
		return "", err
	}
	defer f.Close()
	kmods, err = loadKernelModuleMapping(f)
	if err != nil {
		return "", err
	}

	kernelModules.kmods = kmods
	return kmods[fn], nil
}

// FlushKernelModuleCache removes any cached information about function to kernel module mapping.
func FlushKernelModuleCache() {
	kernelModules.Lock()
	defer kernelModules.Unlock()

	kernelModules.kmods = nil
}

func loadKernelModuleMapping(f io.Reader) (map[string]string, error) {
	mods := make(map[string]string)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := bytes.Fields(scanner.Bytes())
		if len(fields) < 4 {
			continue
		}
		switch string(fields[1]) {
		case "t", "T":
			mods[string(fields[2])] = string(bytes.Trim(fields[3], "[]"))
		default:
			continue
		}
	}
	if scanner.Err() != nil {
		return nil, scanner.Err()
	}
	return mods, nil
}
