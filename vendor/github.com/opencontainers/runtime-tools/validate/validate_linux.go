// +build linux

package validate

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/syndtr/gocapability/capability"

	multierror "github.com/hashicorp/go-multierror"
	rspec "github.com/opencontainers/runtime-spec/specs-go"
	osFilepath "github.com/opencontainers/runtime-tools/filepath"
	"github.com/opencontainers/runtime-tools/specerror"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/sirupsen/logrus"
)

// LastCap return last cap of system
func LastCap() capability.Cap {
	last := capability.CAP_LAST_CAP
	// hack for RHEL6 which has no /proc/sys/kernel/cap_last_cap
	if last == capability.Cap(63) {
		last = capability.CAP_BLOCK_SUSPEND
	}

	return last
}

func deviceValid(d rspec.LinuxDevice) bool {
	switch d.Type {
	case "b", "c", "u":
		if d.Major <= 0 || d.Minor <= 0 {
			return false
		}
	case "p":
		if d.Major != 0 || d.Minor != 0 {
			return false
		}
	default:
		return false
	}
	return true
}

// CheckLinux checks v.spec.Linux
func (v *Validator) CheckLinux() (errs error) {
	logrus.Debugf("check linux")

	if v.spec.Linux == nil {
		return
	}

	var nsTypeList = map[rspec.LinuxNamespaceType]struct {
		num      int
		newExist bool
	}{
		rspec.PIDNamespace:     {0, false},
		rspec.NetworkNamespace: {0, false},
		rspec.MountNamespace:   {0, false},
		rspec.IPCNamespace:     {0, false},
		rspec.UTSNamespace:     {0, false},
		rspec.UserNamespace:    {0, false},
		rspec.CgroupNamespace:  {0, false},
	}

	for index := 0; index < len(v.spec.Linux.Namespaces); index++ {
		ns := v.spec.Linux.Namespaces[index]
		if ns.Path != "" && !osFilepath.IsAbs(v.platform, ns.Path) {
			errs = multierror.Append(errs, specerror.NewError(specerror.NSPathAbs, fmt.Errorf("namespace.path %q is not an absolute path", ns.Path), rspec.Version))
		}

		tmpItem := nsTypeList[ns.Type]
		tmpItem.num = tmpItem.num + 1
		if tmpItem.num > 1 {
			errs = multierror.Append(errs, specerror.NewError(specerror.NSErrorOnDup, fmt.Errorf("duplicated namespace %q", ns.Type), rspec.Version))
		}

		if len(ns.Path) == 0 {
			tmpItem.newExist = true
		}
		nsTypeList[ns.Type] = tmpItem
	}

	if (len(v.spec.Linux.UIDMappings) > 0 || len(v.spec.Linux.GIDMappings) > 0) && !nsTypeList[rspec.UserNamespace].newExist {
		errs = multierror.Append(errs, errors.New("the UID/GID mappings requires a new User namespace to be specified as well"))
	}

	for k := range v.spec.Linux.Sysctl {
		if strings.HasPrefix(k, "net.") && !nsTypeList[rspec.NetworkNamespace].newExist {
			errs = multierror.Append(errs, fmt.Errorf("sysctl %v requires a new Network namespace to be specified as well", k))
		}
		if strings.HasPrefix(k, "fs.mqueue.") {
			if !nsTypeList[rspec.MountNamespace].newExist || !nsTypeList[rspec.IPCNamespace].newExist {
				errs = multierror.Append(errs, fmt.Errorf("sysctl %v requires a new IPC namespace and Mount namespace to be specified as well", k))
			}
		}
	}

	if v.platform == "linux" && !nsTypeList[rspec.UTSNamespace].newExist && v.spec.Hostname != "" {
		errs = multierror.Append(errs, fmt.Errorf("on Linux, hostname requires a new UTS namespace to be specified as well"))
	}

	// Linux devices validation
	devList := make(map[string]bool)
	devTypeList := make(map[string]bool)
	for index := 0; index < len(v.spec.Linux.Devices); index++ {
		device := v.spec.Linux.Devices[index]
		if !deviceValid(device) {
			errs = multierror.Append(errs, fmt.Errorf("device %v is invalid", device))
		}

		if _, exists := devList[device.Path]; exists {
			errs = multierror.Append(errs, fmt.Errorf("device %s is duplicated", device.Path))
		} else {
			var rootfsPath string
			if filepath.IsAbs(v.spec.Root.Path) {
				rootfsPath = v.spec.Root.Path
			} else {
				rootfsPath = filepath.Join(v.bundlePath, v.spec.Root.Path)
			}
			absPath := filepath.Join(rootfsPath, device.Path)
			fi, err := os.Stat(absPath)
			if os.IsNotExist(err) {
				devList[device.Path] = true
			} else if err != nil {
				errs = multierror.Append(errs, err)
			} else {
				fStat, ok := fi.Sys().(*syscall.Stat_t)
				if !ok {
					errs = multierror.Append(errs, specerror.NewError(specerror.DevicesAvailable,
						fmt.Errorf("cannot determine state for device %s", device.Path), rspec.Version))
					continue
				}
				var devType string
				switch fStat.Mode & syscall.S_IFMT {
				case syscall.S_IFCHR:
					devType = "c"
				case syscall.S_IFBLK:
					devType = "b"
				case syscall.S_IFIFO:
					devType = "p"
				default:
					devType = "unmatched"
				}
				if devType != device.Type || (devType == "c" && device.Type == "u") {
					errs = multierror.Append(errs, specerror.NewError(specerror.DevicesFileNotMatch,
						fmt.Errorf("unmatched %s already exists in filesystem", device.Path), rspec.Version))
					continue
				}
				if devType != "p" {
					dev := fStat.Rdev
					major := (dev >> 8) & 0xfff
					minor := (dev & 0xff) | ((dev >> 12) & 0xfff00)
					if int64(major) != device.Major || int64(minor) != device.Minor {
						errs = multierror.Append(errs, specerror.NewError(specerror.DevicesFileNotMatch,
							fmt.Errorf("unmatched %s already exists in filesystem", device.Path), rspec.Version))
						continue
					}
				}
				if device.FileMode != nil {
					expectedPerm := *device.FileMode & os.ModePerm
					actualPerm := fi.Mode() & os.ModePerm
					if expectedPerm != actualPerm {
						errs = multierror.Append(errs, specerror.NewError(specerror.DevicesFileNotMatch,
							fmt.Errorf("unmatched %s already exists in filesystem", device.Path), rspec.Version))
						continue
					}
				}
				if device.UID != nil {
					if *device.UID != fStat.Uid {
						errs = multierror.Append(errs, specerror.NewError(specerror.DevicesFileNotMatch,
							fmt.Errorf("unmatched %s already exists in filesystem", device.Path), rspec.Version))
						continue
					}
				}
				if device.GID != nil {
					if *device.GID != fStat.Gid {
						errs = multierror.Append(errs, specerror.NewError(specerror.DevicesFileNotMatch,
							fmt.Errorf("unmatched %s already exists in filesystem", device.Path), rspec.Version))
						continue
					}
				}
			}
		}

		// unify u->c when comparing, they are synonyms
		var devID string
		if device.Type == "u" {
			devID = fmt.Sprintf("%s:%d:%d", "c", device.Major, device.Minor)
		} else {
			devID = fmt.Sprintf("%s:%d:%d", device.Type, device.Major, device.Minor)
		}

		if _, exists := devTypeList[devID]; exists {
			logrus.Warnf("%v", specerror.NewError(specerror.DevicesErrorOnDup, fmt.Errorf("type:%s, major:%d and minor:%d for linux devices is duplicated", device.Type, device.Major, device.Minor), rspec.Version))
		} else {
			devTypeList[devID] = true
		}
	}

	if v.spec.Linux.Resources != nil {
		errs = multierror.Append(errs, v.CheckLinuxResources())
	}

	for _, maskedPath := range v.spec.Linux.MaskedPaths {
		if !strings.HasPrefix(maskedPath, "/") {
			errs = multierror.Append(errs,
				specerror.NewError(
					specerror.MaskedPathsAbs,
					fmt.Errorf("maskedPath %v is not an absolute path", maskedPath),
					rspec.Version))
		}
	}

	for _, readonlyPath := range v.spec.Linux.ReadonlyPaths {
		if !strings.HasPrefix(readonlyPath, "/") {
			errs = multierror.Append(errs,
				specerror.NewError(
					specerror.ReadonlyPathsAbs,
					fmt.Errorf("readonlyPath %v is not an absolute path", readonlyPath),
					rspec.Version))
		}
	}

	if v.spec.Linux.MountLabel != "" {
		if err := label.Validate(v.spec.Linux.MountLabel); err != nil {
			errs = multierror.Append(errs, fmt.Errorf("mountLabel %v is invalid", v.spec.Linux.MountLabel))
		}
	}

	return
}
