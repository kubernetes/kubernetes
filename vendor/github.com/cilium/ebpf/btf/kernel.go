package btf

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/kallsyms"
)

var kernelBTF = struct {
	sync.RWMutex
	kernel  *Spec
	modules map[string]*Spec
}{
	modules: make(map[string]*Spec),
}

// FlushKernelSpec removes any cached kernel type information.
func FlushKernelSpec() {
	kallsyms.FlushKernelModuleCache()

	kernelBTF.Lock()
	defer kernelBTF.Unlock()

	kernelBTF.kernel = nil
	kernelBTF.modules = make(map[string]*Spec)
}

// LoadKernelSpec returns the current kernel's BTF information.
//
// Defaults to /sys/kernel/btf/vmlinux and falls back to scanning the file system
// for vmlinux ELFs. Returns an error wrapping ErrNotSupported if BTF is not enabled.
func LoadKernelSpec() (*Spec, error) {
	kernelBTF.RLock()
	spec := kernelBTF.kernel
	kernelBTF.RUnlock()

	if spec == nil {
		kernelBTF.Lock()
		defer kernelBTF.Unlock()

		spec = kernelBTF.kernel
	}

	if spec != nil {
		return spec.Copy(), nil
	}

	spec, _, err := loadKernelSpec()
	if err != nil {
		return nil, err
	}

	kernelBTF.kernel = spec
	return spec.Copy(), nil
}

// LoadKernelModuleSpec returns the BTF information for the named kernel module.
//
// Defaults to /sys/kernel/btf/<module>.
// Returns an error wrapping ErrNotSupported if BTF is not enabled.
// Returns an error wrapping fs.ErrNotExist if BTF for the specific module doesn't exist.
func LoadKernelModuleSpec(module string) (*Spec, error) {
	kernelBTF.RLock()
	spec := kernelBTF.modules[module]
	kernelBTF.RUnlock()

	if spec != nil {
		return spec.Copy(), nil
	}

	base, err := LoadKernelSpec()
	if err != nil {
		return nil, fmt.Errorf("load kernel spec: %w", err)
	}

	kernelBTF.Lock()
	defer kernelBTF.Unlock()

	if spec = kernelBTF.modules[module]; spec != nil {
		return spec.Copy(), nil
	}

	spec, err = loadKernelModuleSpec(module, base)
	if err != nil {
		return nil, err
	}

	kernelBTF.modules[module] = spec
	return spec.Copy(), nil
}

func loadKernelSpec() (_ *Spec, fallback bool, _ error) {
	fh, err := os.Open("/sys/kernel/btf/vmlinux")
	if err == nil {
		defer fh.Close()

		spec, err := loadRawSpec(fh, internal.NativeEndian, nil)
		return spec, false, err
	}

	file, err := findVMLinux()
	if err != nil {
		return nil, false, err
	}
	defer file.Close()

	spec, err := LoadSpecFromReader(file)
	return spec, true, err
}

func loadKernelModuleSpec(module string, base *Spec) (*Spec, error) {
	dir, file := filepath.Split(module)
	if dir != "" || filepath.Ext(file) != "" {
		return nil, fmt.Errorf("invalid module name %q", module)
	}

	fh, err := os.Open(filepath.Join("/sys/kernel/btf", module))
	if err != nil {
		return nil, err
	}
	defer fh.Close()

	return loadRawSpec(fh, internal.NativeEndian, base)
}

// findVMLinux scans multiple well-known paths for vmlinux kernel images.
func findVMLinux() (*os.File, error) {
	release, err := internal.KernelRelease()
	if err != nil {
		return nil, err
	}

	// use same list of locations as libbpf
	// https://github.com/libbpf/libbpf/blob/9a3a42608dbe3731256a5682a125ac1e23bced8f/src/btf.c#L3114-L3122
	locations := []string{
		"/boot/vmlinux-%s",
		"/lib/modules/%s/vmlinux-%[1]s",
		"/lib/modules/%s/build/vmlinux",
		"/usr/lib/modules/%s/kernel/vmlinux",
		"/usr/lib/debug/boot/vmlinux-%s",
		"/usr/lib/debug/boot/vmlinux-%s.debug",
		"/usr/lib/debug/lib/modules/%s/vmlinux",
	}

	for _, loc := range locations {
		file, err := os.Open(fmt.Sprintf(loc, release))
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		return file, err
	}

	return nil, fmt.Errorf("no BTF found for kernel version %s: %w", release, internal.ErrNotSupported)
}
