package label

import (
	"errors"
	"fmt"
	"strings"

	"github.com/opencontainers/selinux/go-selinux"
)

// Valid Label Options
var validOptions = map[string]bool{
	"disable":  true,
	"type":     true,
	"filetype": true,
	"user":     true,
	"role":     true,
	"level":    true,
}

var ErrIncompatibleLabel = errors.New("Bad SELinux option z and Z can not be used together")

// InitLabels returns the process label and file labels to be used within
// the container.  A list of options can be passed into this function to alter
// the labels.  The labels returned will include a random MCS String, that is
// guaranteed to be unique.
// If the disabled flag is passed in, the process label will not be set, but the mount label will be set
// to the container_file label with the maximum category. This label is not usable by any confined label.
func InitLabels(options []string) (plabel string, mlabel string, retErr error) {
	if !selinux.GetEnabled() {
		return "", "", nil
	}
	processLabel, mountLabel := selinux.ContainerLabels()
	if processLabel != "" {
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
			if opt == "disable" {
				selinux.ReleaseLabel(mountLabel)
				return "", selinux.PrivContainerMountLabel(), nil
			}
			if i := strings.Index(opt, ":"); i == -1 {
				return "", "", fmt.Errorf("Bad label option %q, valid options 'disable' or \n'user, role, level, type, filetype' followed by ':' and a value", opt)
			}
			con := strings.SplitN(opt, ":", 2)
			if !validOptions[con[0]] {
				return "", "", fmt.Errorf("Bad label option %q, valid options 'disable, user, role, level, type, filetype'", con[0])
			}
			if con[0] == "filetype" {
				mcon["type"] = con[1]
				continue
			}
			pcon[con[0]] = con[1]
			if con[0] == "level" || con[0] == "user" {
				mcon[con[0]] = con[1]
			}
		}
		if pcon.Get() != processLabel {
			if pcon["level"] != mcsLevel {
				selinux.ReleaseLabel(processLabel)
			}
			processLabel = pcon.Get()
			selinux.ReserveLabel(processLabel)
		}
		mountLabel = mcon.Get()
	}
	return processLabel, mountLabel, nil
}

// Deprecated: The GenLabels function is only to be used during the transition
// to the official API. Use InitLabels(strings.Fields(options)) instead.
func GenLabels(options string) (string, string, error) {
	return InitLabels(strings.Fields(options))
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
	if err := selinux.Chcon(path, fileLabel, true); err != nil {
		return err
	}
	return nil
}

// DisableSecOpt returns a security opt that can disable labeling
// support for future container processes
// Deprecated: use selinux.DisableSecOpt
var DisableSecOpt = selinux.DisableSecOpt

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
