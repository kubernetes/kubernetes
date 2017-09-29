// +build selinux,linux

package label

import (
	"fmt"
	"strings"

	"github.com/opencontainers/selinux/go-selinux"
)

// Valid Label Options
var validOptions = map[string]bool{
	"disable": true,
	"type":    true,
	"user":    true,
	"role":    true,
	"level":   true,
}

var ErrIncompatibleLabel = fmt.Errorf("Bad SELinux option z and Z can not be used together")

// InitLabels returns the process label and file labels to be used within
// the container.  A list of options can be passed into this function to alter
// the labels.  The labels returned will include a random MCS String, that is
// guaranteed to be unique.
func InitLabels(options []string) (string, string, error) {
	if !selinux.GetEnabled() {
		return "", "", nil
	}
	processLabel, mountLabel := selinux.ContainerLabels()
	if processLabel != "" {
		pcon := selinux.NewContext(processLabel)
		mcon := selinux.NewContext(mountLabel)
		for _, opt := range options {
			if opt == "disable" {
				return "", "", nil
			}
			if i := strings.Index(opt, ":"); i == -1 {
				return "", "", fmt.Errorf("Bad label option %q, valid options 'disable' or \n'user, role, level, type' followed by ':' and a value", opt)
			}
			con := strings.SplitN(opt, ":", 2)
			if !validOptions[con[0]] {
				return "", "", fmt.Errorf("Bad label option %q, valid options 'disable, user, role, level, type'", con[0])

			}
			pcon[con[0]] = con[1]
			if con[0] == "level" || con[0] == "user" {
				mcon[con[0]] = con[1]
			}
		}
		processLabel = pcon.Get()
		mountLabel = mcon.Get()
	}
	return processLabel, mountLabel, nil
}

func ROMountLabel() string {
	return selinux.ROFileLabel()
}

// DEPRECATED: The GenLabels function is only to be used during the transition to the official API.
func GenLabels(options string) (string, string, error) {
	return InitLabels(strings.Fields(options))
}

// FormatMountLabel returns a string to be used by the mount command.
// The format of this string will be used to alter the labeling of the mountpoint.
// The string returned is suitable to be used as the options field of the mount command.
// If you need to have additional mount point options, you can pass them in as
// the first parameter.  Second parameter is the label that you wish to apply
// to all content in the mount point.
func FormatMountLabel(src, mountLabel string) string {
	if mountLabel != "" {
		switch src {
		case "":
			src = fmt.Sprintf("context=%q", mountLabel)
		default:
			src = fmt.Sprintf("%s,context=%q", src, mountLabel)
		}
	}
	return src
}

// SetProcessLabel takes a process label and tells the kernel to assign the
// label to the next program executed by the current process.
func SetProcessLabel(processLabel string) error {
	if processLabel == "" {
		return nil
	}
	return selinux.SetExecLabel(processLabel)
}

// ProcessLabel returns the process label that the kernel will assign
// to the next program executed by the current process.  If "" is returned
// this indicates that the default labeling will happen for the process.
func ProcessLabel() (string, error) {
	return selinux.ExecLabel()
}

// GetFileLabel returns the label for specified path
func FileLabel(path string) (string, error) {
	return selinux.FileLabel(path)
}

// SetFileLabel modifies the "path" label to the specified file label
func SetFileLabel(path string, fileLabel string) error {
	if selinux.GetEnabled() && fileLabel != "" {
		return selinux.SetFileLabel(path, fileLabel)
	}
	return nil
}

// SetFileCreateLabel tells the kernel the label for all files to be created
func SetFileCreateLabel(fileLabel string) error {
	if selinux.GetEnabled() {
		return selinux.SetFSCreateLabel(fileLabel)
	}
	return nil
}

// Relabel changes the label of path to the filelabel string.
// It changes the MCS label to s0 if shared is true.
// This will allow all containers to share the content.
func Relabel(path string, fileLabel string, shared bool) error {
	if !selinux.GetEnabled() {
		return nil
	}

	if fileLabel == "" {
		return nil
	}

	exclude_paths := map[string]bool{"/": true, "/usr": true, "/etc": true}
	if exclude_paths[path] {
		return fmt.Errorf("SELinux relabeling of %s is not allowed", path)
	}

	if shared {
		c := selinux.NewContext(fileLabel)
		c["level"] = "s0"
		fileLabel = c.Get()
	}
	if err := selinux.Chcon(path, fileLabel, true); err != nil {
		return err
	}
	return nil
}

// PidLabel will return the label of the process running with the specified pid
func PidLabel(pid int) (string, error) {
	return selinux.PidLabel(pid)
}

// Init initialises the labeling system
func Init() {
	selinux.GetEnabled()
}

// ReserveLabel will record the fact that the MCS label has already been used.
// This will prevent InitLabels from using the MCS label in a newly created
// container
func ReserveLabel(label string) error {
	selinux.ReserveLabel(label)
	return nil
}

// ReleaseLabel will remove the reservation of the MCS label.
// This will allow InitLabels to use the MCS label in a newly created
// containers
func ReleaseLabel(label string) error {
	selinux.ReleaseLabel(label)
	return nil
}

// DupSecOpt takes a process label and returns security options that
// can be used to set duplicate labels on future container processes
func DupSecOpt(src string) []string {
	return selinux.DupSecOpt(src)
}

// DisableSecOpt returns a security opt that can disable labeling
// support for future container processes
func DisableSecOpt() []string {
	return selinux.DisableSecOpt()
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
