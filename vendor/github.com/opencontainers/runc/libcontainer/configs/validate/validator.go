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
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

type Validator interface {
	Validate(*configs.Config) error
}

func New() Validator {
	return &ConfigValidator{}
}

type ConfigValidator struct{}

type check func(config *configs.Config) error

func (v *ConfigValidator) Validate(config *configs.Config) error {
	checks := []check{
		v.cgroups,
		v.rootfs,
		v.network,
		v.hostname,
		v.security,
		v.usernamespace,
		v.cgroupnamespace,
		v.sysctl,
		v.intelrdt,
		v.rootlessEUID,
	}
	for _, c := range checks {
		if err := c(config); err != nil {
			return err
		}
	}
	// Relaxed validation rules for backward compatibility
	warns := []check{
		v.mounts, // TODO (runc v1.x.x): make this an error instead of a warning
	}
	for _, c := range warns {
		if err := c(config); err != nil {
			logrus.WithError(err).Warn("invalid configuration")
		}
	}
	return nil
}

// rootfs validates if the rootfs is an absolute path and is not a symlink
// to the container's root filesystem.
func (v *ConfigValidator) rootfs(config *configs.Config) error {
	if _, err := os.Stat(config.Rootfs); err != nil {
		return fmt.Errorf("invalid rootfs: %w", err)
	}
	cleaned, err := filepath.Abs(config.Rootfs)
	if err != nil {
		return fmt.Errorf("invalid rootfs: %w", err)
	}
	if cleaned, err = filepath.EvalSymlinks(cleaned); err != nil {
		return fmt.Errorf("invalid rootfs: %w", err)
	}
	if filepath.Clean(config.Rootfs) != cleaned {
		return errors.New("invalid rootfs: not an absolute path, or a symlink")
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
			return errors.New("user namespaces aren't enabled in the kernel")
		}
		hasPath := config.Namespaces.PathOf(configs.NEWUSER) != ""
		hasMappings := config.UidMappings != nil || config.GidMappings != nil
		if !hasPath && !hasMappings {
			return errors.New("user namespaces enabled, but no namespace path to join nor mappings to apply specified")
		}
		// The hasPath && hasMappings validation case is handled in specconv --
		// we cache the mappings in Config during specconv in the hasPath case,
		// so we cannot do that validation here.
	} else {
		if config.UidMappings != nil || config.GidMappings != nil {
			return errors.New("user namespace mappings specified, but user namespace isn't enabled in the config")
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

// convertSysctlVariableToDotsSeparator can return sysctl variables in dots separator format.
// The '/' separator is also accepted in place of a '.'.
// Convert the sysctl variables to dots separator format for validation.
// More info: sysctl(8), sysctl.d(5).
//
// For example:
// Input sysctl variable "net/ipv4/conf/eno2.100.rp_filter"
// will return the converted value "net.ipv4.conf.eno2/100.rp_filter"
func convertSysctlVariableToDotsSeparator(val string) string {
	if val == "" {
		return val
	}
	firstSepIndex := strings.IndexAny(val, "./")
	if firstSepIndex == -1 || val[firstSepIndex] == '.' {
		return val
	}

	f := func(r rune) rune {
		switch r {
		case '.':
			return '/'
		case '/':
			return '.'
		}
		return r
	}
	return strings.Map(f, val)
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
		s := convertSysctlVariableToDotsSeparator(s)
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
				return fmt.Errorf("invalid netns path: %w", hostnetErr)
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
		if config.IntelRdt.ClosID == "." || config.IntelRdt.ClosID == ".." || strings.Contains(config.IntelRdt.ClosID, "/") {
			return fmt.Errorf("invalid intelRdt.ClosID %q", config.IntelRdt.ClosID)
		}

		if !intelrdt.IsCATEnabled() && config.IntelRdt.L3CacheSchema != "" {
			return errors.New("intelRdt.l3CacheSchema is specified in config, but Intel RDT/CAT is not enabled")
		}
		if !intelrdt.IsMBAEnabled() && config.IntelRdt.MemBwSchema != "" {
			return errors.New("intelRdt.memBwSchema is specified in config, but Intel RDT/MBA is not enabled")
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
		return false, &os.PathError{Op: "stat", Path: currentProcessNetns, Err: err}
	}
	if err := unix.Stat(path, &st2); err != nil {
		return false, &os.PathError{Op: "stat", Path: path, Err: err}
	}

	return (st1.Dev == st2.Dev) && (st1.Ino == st2.Ino), nil
}
