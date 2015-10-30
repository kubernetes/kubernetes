// +build selinux,linux

package label

import (
	"fmt"
	"strings"

	"github.com/docker/libcontainer/selinux"
)

// InitLabels returns the process label and file labels to be used within
// the container.  A list of options can be passed into this function to alter
// the labels.  The labels returned will include a random MCS String, that is
// guaranteed to be unique.
func InitLabels(options []string) (string, string, error) {
	if !selinux.SelinuxEnabled() {
		return "", "", nil
	}
	processLabel, mountLabel := selinux.GetLxcContexts()
	if processLabel != "" {
		pcon := selinux.NewContext(processLabel)
		mcon := selinux.NewContext(mountLabel)
		for _, opt := range options {
			if opt == "disable" {
				return "", "", nil
			}
			if i := strings.Index(opt, ":"); i == -1 {
				return "", "", fmt.Errorf("Bad SELinux Option")
			}
			con := strings.SplitN(opt, ":", 2)
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
	return selinux.Setexeccon(processLabel)
}

// GetProcessLabel returns the process label that the kernel will assign
// to the next program executed by the current process.  If "" is returned
// this indicates that the default labeling will happen for the process.
func GetProcessLabel() (string, error) {
	return selinux.Getexeccon()
}

// SetFileLabel modifies the "path" label to the specified file label
func SetFileLabel(path string, fileLabel string) error {
	if selinux.SelinuxEnabled() && fileLabel != "" {
		return selinux.Setfilecon(path, fileLabel)
	}
	return nil
}

// Tell the kernel the label for all files to be created
func SetFileCreateLabel(fileLabel string) error {
	if selinux.SelinuxEnabled() {
		return selinux.Setfscreatecon(fileLabel)
	}
	return nil
}

// Change the label of path to the filelabel string.  If the relabel string
// is "z", relabel will change the MCS label to s0.  This will allow all
// containers to share the content.  If the relabel string is a "Z" then
// the MCS label should continue to be used.  SELinux will use this field
// to make sure the content can not be shared by other containes.
func Relabel(path string, fileLabel string, relabel string) error {
	exclude_path := []string{"/", "/usr", "/etc"}
	if fileLabel == "" {
		return nil
	}
	if !strings.ContainsAny(relabel, "zZ") {
		return nil
	}
	for _, p := range exclude_path {
		if path == p {
			return fmt.Errorf("Relabeling of %s is not allowed", path)
		}
	}
	if strings.Contains(relabel, "z") && strings.Contains(relabel, "Z") {
		return fmt.Errorf("Bad SELinux option z and Z can not be used together")
	}
	if strings.Contains(relabel, "z") {
		c := selinux.NewContext(fileLabel)
		c["level"] = "s0"
		fileLabel = c.Get()
	}
	return selinux.Chcon(path, fileLabel, true)
}

// GetPidLabel will return the label of the process running with the specified pid
func GetPidLabel(pid int) (string, error) {
	return selinux.Getpidcon(pid)
}

// Init initialises the labeling system
func Init() {
	selinux.SelinuxEnabled()
}

// ReserveLabel will record the fact that the MCS label has already been used.
// This will prevent InitLabels from using the MCS label in a newly created
// container
func ReserveLabel(label string) error {
	selinux.ReserveLabel(label)
	return nil
}

// UnreserveLabel will remove the reservation of the MCS label.
// This will allow InitLabels to use the MCS label in a newly created
// containers
func UnreserveLabel(label string) error {
	selinux.FreeLxcContexts(label)
	return nil
}

// DupSecOpt takes an process label and returns security options that
// can be used to set duplicate labels on future container processes
func DupSecOpt(src string) []string {
	return selinux.DupSecOpt(src)
}

// DisableSecOpt returns a security opt that can disable labeling
// support for future container processes
func DisableSecOpt() []string {
	return selinux.DisableSecOpt()
}
