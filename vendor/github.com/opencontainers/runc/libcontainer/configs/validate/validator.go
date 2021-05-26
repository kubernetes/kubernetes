package validate

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/intelrdt"
	selinux "github.com/opencontainers/selinux/go-selinux"
	"golang.org/x/sys/unix"
)

type Validator interface {
	Validate(*configs.Config) error
}

func New() Validator {
	return &ConfigValidator{}
}

type ConfigValidator struct {
}

type check func(config *configs.Config) error

func (v *ConfigValidator) Validate(config *configs.Config) error {
	checks := []check{
		v.rootfs,
		v.network,
		v.hostname,
		v.security,
		v.usernamespace,
		v.cgroupnamespace,
		v.sysctl,
		v.intelrdt,
		v.rootlessEUID,
		v.mounts,
	}
	for _, c := range checks {
		if err := c(config); err != nil {
			return err
		}
	}
	if err := v.cgroups(config); err != nil {
		return err
	}

	return nil
}

// rootfs validates if the rootfs is an absolute path and is not a symlink
// to the container's root filesystem.
func (v *ConfigValidator) rootfs(config *configs.Config) error {
	if _, err := os.Stat(config.Rootfs); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("rootfs (%s) does not exist", config.Rootfs)
		}
		return err
	}
	cleaned, err := filepath.Abs(config.Rootfs)
	if err != nil {
		return err
	}
	if cleaned, err = filepath.EvalSymlinks(cleaned); err != nil {
		return err
	}
	if filepath.Clean(config.Rootfs) != cleaned {
		return fmt.Errorf("%s is not an absolute path or is a symlink", config.Rootfs)
	}
	return nil
}

func (v *ConfigValidator) network(config *configs.Config) error {
	if !config.Namespaces.Contains(configs.NEWNET) {
		if len(config.Networks) > 0 || len(config.Routes) > 0 {
			return errors.New("unable to apply network settings without a private NET namespace")
		}
	}
	return nil
}

func (v *ConfigValidator) hostname(config *configs.Config) error {
	if config.Hostname != "" && !config.Namespaces.Contains(configs.NEWUTS) {
		return errors.New("unable to set hostname without a private UTS namespace")
	}
	return nil
}

func (v *ConfigValidator) security(config *configs.Config) error {
	// restrict sys without mount namespace
	if (len(config.MaskPaths) > 0 || len(config.ReadonlyPaths) > 0) &&
		!config.Namespaces.Contains(configs.NEWNS) {
		return errors.New("unable to restrict sys entries without a private MNT namespace")
	}
	if config.ProcessLabel != "" && !selinux.GetEnabled() {
		return errors.New("selinux label is specified in config, but selinux is disabled or not supported")
	}

	return nil
}

func (v *ConfigValidator) usernamespace(config *configs.Config) error {
	if config.Namespaces.Contains(configs.NEWUSER) {
		if _, err := os.Stat("/proc/self/ns/user"); os.IsNotExist(err) {
			return errors.New("USER namespaces aren't enabled in the kernel")
		}
	} else {
		if config.UidMappings != nil || config.GidMappings != nil {
			return errors.New("User namespace mappings specified, but USER namespace isn't enabled in the config")
		}
	}
	return nil
}

func (v *ConfigValidator) cgroupnamespace(config *configs.Config) error {
	if config.Namespaces.Contains(configs.NEWCGROUP) {
		if _, err := os.Stat("/proc/self/ns/cgroup"); os.IsNotExist(err) {
			return errors.New("cgroup namespaces aren't enabled in the kernel")
		}
	}
	return nil
}

// sysctl validates that the specified sysctl keys are valid or not.
// /proc/sys isn't completely namespaced and depending on which namespaces
// are specified, a subset of sysctls are permitted.
func (v *ConfigValidator) sysctl(config *configs.Config) error {
	validSysctlMap := map[string]bool{
		"kernel.msgmax":          true,
		"kernel.msgmnb":          true,
		"kernel.msgmni":          true,
		"kernel.sem":             true,
		"kernel.shmall":          true,
		"kernel.shmmax":          true,
		"kernel.shmmni":          true,
		"kernel.shm_rmid_forced": true,
	}

	var (
		netOnce    sync.Once
		hostnet    bool
		hostnetErr error
	)

	for s := range config.Sysctl {
		if validSysctlMap[s] || strings.HasPrefix(s, "fs.mqueue.") {
			if config.Namespaces.Contains(configs.NEWIPC) {
				continue
			} else {
				return fmt.Errorf("sysctl %q is not allowed in the hosts ipc namespace", s)
			}
		}
		if strings.HasPrefix(s, "net.") {
			// Is container using host netns?
			// Here "host" means "current", not "initial".
			netOnce.Do(func() {
				if !config.Namespaces.Contains(configs.NEWNET) {
					hostnet = true
					return
				}
				path := config.Namespaces.PathOf(configs.NEWNET)
				if path == "" {
					// own netns, so hostnet = false
					return
				}
				hostnet, hostnetErr = isHostNetNS(path)
			})
			if hostnetErr != nil {
				return hostnetErr
			}
			if hostnet {
				return fmt.Errorf("sysctl %q not allowed in host network namespace", s)
			}
			continue
		}
		if config.Namespaces.Contains(configs.NEWUTS) {
			switch s {
			case "kernel.domainname":
				// This is namespaced and there's no explicit OCI field for it.
				continue
			case "kernel.hostname":
				// This is namespaced but there's a conflicting (dedicated) OCI field for it.
				return fmt.Errorf("sysctl %q is not allowed as it conflicts with the OCI %q field", s, "hostname")
			}
		}
		return fmt.Errorf("sysctl %q is not in a separate kernel namespace", s)
	}

	return nil
}

func (v *ConfigValidator) intelrdt(config *configs.Config) error {
	if config.IntelRdt != nil {
		if !intelrdt.IsCATEnabled() && !intelrdt.IsMBAEnabled() {
			return errors.New("intelRdt is specified in config, but Intel RDT is not supported or enabled")
		}

		if !intelrdt.IsCATEnabled() && config.IntelRdt.L3CacheSchema != "" {
			return errors.New("intelRdt.l3CacheSchema is specified in config, but Intel RDT/CAT is not enabled")
		}
		if !intelrdt.IsMBAEnabled() && config.IntelRdt.MemBwSchema != "" {
			return errors.New("intelRdt.memBwSchema is specified in config, but Intel RDT/MBA is not enabled")
		}

		if intelrdt.IsCATEnabled() && config.IntelRdt.L3CacheSchema == "" {
			return errors.New("Intel RDT/CAT is enabled and intelRdt is specified in config, but intelRdt.l3CacheSchema is empty")
		}
		if intelrdt.IsMBAEnabled() && config.IntelRdt.MemBwSchema == "" {
			return errors.New("Intel RDT/MBA is enabled and intelRdt is specified in config, but intelRdt.memBwSchema is empty")
		}
	}

	return nil
}

func (v *ConfigValidator) cgroups(config *configs.Config) error {
	c := config.Cgroups
	if c == nil {
		return nil
	}

	if (c.Name != "" || c.Parent != "") && c.Path != "" {
		return fmt.Errorf("cgroup: either Path or Name and Parent should be used, got %+v", c)
	}

	r := c.Resources
	if r == nil {
		return nil
	}

	if !cgroups.IsCgroup2UnifiedMode() && r.Unified != nil {
		return cgroups.ErrV1NoUnified
	}

	if cgroups.IsCgroup2UnifiedMode() {
		_, err := cgroups.ConvertMemorySwapToCgroupV2Value(r.MemorySwap, r.Memory)
		if err != nil {
			return err
		}
	}

	return nil
}

func (v *ConfigValidator) mounts(config *configs.Config) error {
	for _, m := range config.Mounts {
		if !filepath.IsAbs(m.Destination) {
			return fmt.Errorf("invalid mount %+v: mount destination not absolute", m)
		}
	}

	return nil
}

func isHostNetNS(path string) (bool, error) {
	const currentProcessNetns = "/proc/self/ns/net"

	var st1, st2 unix.Stat_t

	if err := unix.Stat(currentProcessNetns, &st1); err != nil {
		return false, fmt.Errorf("unable to stat %q: %s", currentProcessNetns, err)
	}
	if err := unix.Stat(path, &st2); err != nil {
		return false, fmt.Errorf("unable to stat %q: %s", path, err)
	}

	return (st1.Dev == st2.Dev) && (st1.Ino == st2.Ino), nil
}
