package volume

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/docker/docker/api/types/mount"
)

var errBindNotExist = errors.New("bind source path does not exist")

type validateOpts struct {
	skipBindSourceCheck   bool
	skipAbsolutePathCheck bool
}

func validateMountConfig(mnt *mount.Mount, options ...func(*validateOpts)) error {
	opts := validateOpts{}
	for _, o := range options {
		o(&opts)
	}

	if len(mnt.Target) == 0 {
		return &errMountConfig{mnt, errMissingField("Target")}
	}

	if err := validateNotRoot(mnt.Target); err != nil {
		return &errMountConfig{mnt, err}
	}

	if !opts.skipAbsolutePathCheck {
		if err := validateAbsolute(mnt.Target); err != nil {
			return &errMountConfig{mnt, err}
		}
	}

	switch mnt.Type {
	case mount.TypeBind:
		if len(mnt.Source) == 0 {
			return &errMountConfig{mnt, errMissingField("Source")}
		}
		// Don't error out just because the propagation mode is not supported on the platform
		if opts := mnt.BindOptions; opts != nil {
			if len(opts.Propagation) > 0 && len(propagationModes) > 0 {
				if _, ok := propagationModes[opts.Propagation]; !ok {
					return &errMountConfig{mnt, fmt.Errorf("invalid propagation mode: %s", opts.Propagation)}
				}
			}
		}
		if mnt.VolumeOptions != nil {
			return &errMountConfig{mnt, errExtraField("VolumeOptions")}
		}

		if err := validateAbsolute(mnt.Source); err != nil {
			return &errMountConfig{mnt, err}
		}

		// Do not allow binding to non-existent path
		if !opts.skipBindSourceCheck {
			fi, err := os.Stat(mnt.Source)
			if err != nil {
				if !os.IsNotExist(err) {
					return &errMountConfig{mnt, err}
				}
				return &errMountConfig{mnt, errBindNotExist}
			}
			if err := validateStat(fi); err != nil {
				return &errMountConfig{mnt, err}
			}
		}
	case mount.TypeVolume:
		if mnt.BindOptions != nil {
			return &errMountConfig{mnt, errExtraField("BindOptions")}
		}

		if len(mnt.Source) == 0 && mnt.ReadOnly {
			return &errMountConfig{mnt, fmt.Errorf("must not set ReadOnly mode when using anonymous volumes")}
		}

		if len(mnt.Source) != 0 {
			if valid, err := IsVolumeNameValid(mnt.Source); !valid {
				if err == nil {
					err = errors.New("invalid volume name")
				}
				return &errMountConfig{mnt, err}
			}
		}
	case mount.TypeTmpfs:
		if len(mnt.Source) != 0 {
			return &errMountConfig{mnt, errExtraField("Source")}
		}
		if err := ValidateTmpfsMountDestination(mnt.Target); err != nil {
			return &errMountConfig{mnt, err}
		}
		if _, err := ConvertTmpfsOptions(mnt.TmpfsOptions, mnt.ReadOnly); err != nil {
			return &errMountConfig{mnt, err}
		}
	default:
		return &errMountConfig{mnt, errors.New("mount type unknown")}
	}
	return nil
}

type errMountConfig struct {
	mount *mount.Mount
	err   error
}

func (e *errMountConfig) Error() string {
	return fmt.Sprintf("invalid mount config for type %q: %v", e.mount.Type, e.err.Error())
}

func errExtraField(name string) error {
	return fmt.Errorf("field %s must not be specified", name)
}
func errMissingField(name string) error {
	return fmt.Errorf("field %s must not be empty", name)
}

func validateAbsolute(p string) error {
	p = convertSlash(p)
	if filepath.IsAbs(p) {
		return nil
	}
	return fmt.Errorf("invalid mount path: '%s' mount path must be absolute", p)
}

// ValidateTmpfsMountDestination validates the destination of tmpfs mount.
// Currently, we have only two obvious rule for validation:
//  - path must not be "/"
//  - path must be absolute
// We should add more rules carefully (#30166)
func ValidateTmpfsMountDestination(dest string) error {
	if err := validateNotRoot(dest); err != nil {
		return err
	}
	return validateAbsolute(dest)
}
