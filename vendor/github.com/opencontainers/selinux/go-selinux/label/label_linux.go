package label

import (
	"errors"
	"fmt"
	"strings"

	"github.com/opencontainers/selinux/go-selinux"
)

// Valid Label Options
var validOptions = map[string]bool{
	"type":     true,
	"filetype": true,
	"user":     true,
	"role":     true,
	"level":    true,
}

var ErrIncompatibleLabel = errors.New("bad SELinux option: z and Z can not be used together")

// InitLabels returns the process label and file labels to be used within
// the container.  A list of options can be passed into this function to alter
// the labels.
//
// Unless the "level" option is provided (to set a custom level), the labels
// returned will include a random MCS string guaranteed to be unique in the
// scope of the process using this package. If the "level" option is provided,
// the custom level set is reserved but not checked to be unique.
//
// If the disabled flag is passed in, the process label will not be set, but the mount label will be set
// to the container_file label with the maximum category. This label is not usable by any confined label.
func InitLabels(options []string) (plabel string, mlabel string, retErr error) {
	if !selinux.GetEnabled() {
		return "", "", nil
	}
	if len(options) > 0 && options[0] == "disable" {
		return "", selinux.PrivContainerMountLabel(), nil
	}
	processLabel, mountLabel := selinux.ContainerLabels() //nolint:staticcheck // ContainerLabels will be moved to an internal package.
	if processLabel == "" || len(options) == 0 {
		// 1. processLabel is required; if empty, do nothing.
		// 2. If there are no options to process, we're done.
		return processLabel, mountLabel, nil
	}
	defer func() {
		if retErr != nil {
			selinux.ReleaseLabel(mountLabel)
		}
	}()
	pcon, err := selinux.NewContext(processLabel)
	if err != nil {
		return "", "", err
	}
	mcsLevel := pcon["level"]
	mcon, err := selinux.NewContext(mountLabel)
	if err != nil {
		return "", "", err
	}
	for _, opt := range options {
		// For backward compatibility, process "disable"
		// even if it's not the only option.
		if opt == "disable" {
			selinux.ReleaseLabel(mountLabel)
			return "", selinux.PrivContainerMountLabel(), nil
		}
		k, v, ok := strings.Cut(opt, ":")
		if !ok || !validOptions[k] {
			return "", "", fmt.Errorf("bad label option %q, valid options 'disable' or \n'user, role, level, type, filetype' followed by ':' and a value", opt)
		}
		if k == "filetype" {
			mcon["type"] = v
			continue
		}
		pcon[k] = v
		if k == "level" || k == "user" {
			mcon[k] = v
		}
	}
	if p := pcon.Get(); p != processLabel {
		if pcon["level"] != mcsLevel {
			selinux.ReleaseLabel(processLabel)
			// Ignore ErrMCSAlreadyExists as label is user-specified and might be
			// already reserved (e.g. when containers in a pod use the same label).
			if err := selinux.ReserveLabelV2(p); err != nil && !errors.Is(err, selinux.ErrMCSAlreadyExists) {
				return "", "", err
			}
		}
		processLabel = p
	}
	mountLabel = mcon.Get()
	return processLabel, mountLabel, nil
}

// SetFileLabel modifies the "path" label to the specified file label
func SetFileLabel(path string, fileLabel string) error {
	if !selinux.GetEnabled() || fileLabel == "" {
		return nil
	}
	return selinux.SetFileLabel(path, fileLabel)
}

// SetFileCreateLabel tells the kernel the label for all files to be created
func SetFileCreateLabel(fileLabel string) error {
	if !selinux.GetEnabled() {
		return nil
	}
	return selinux.SetFSCreateLabel(fileLabel)
}

// Relabel changes the label of path and all the entries beneath the path.
// It changes the MCS label to s0 if shared is true.
// This will allow all containers to share the content.
//
// The path itself is guaranteed to be relabeled last.
func Relabel(path string, fileLabel string, shared bool) error {
	if !selinux.GetEnabled() || fileLabel == "" {
		return nil
	}

	if shared {
		c, err := selinux.NewContext(fileLabel)
		if err != nil {
			return err
		}

		c["level"] = "s0"
		fileLabel = c.Get()
	}
	return selinux.Chcon(path, fileLabel, true)
}

// Validate checks that the label does not include unexpected options
func Validate(label string) error {
	if strings.Contains(label, "z") && strings.Contains(label, "Z") {
		return ErrIncompatibleLabel
	}
	return nil
}

// RelabelNeeded checks whether the user requested a relabel
func RelabelNeeded(label string) bool {
	return strings.Contains(label, "z") || strings.Contains(label, "Z")
}

// IsShared checks that the label includes a "shared" mark
func IsShared(label string) bool {
	return strings.Contains(label, "z")
}
