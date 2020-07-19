package label

import (
	"fmt"

	"github.com/opencontainers/selinux/go-selinux"
)

// Deprecated: use selinux.ROFileLabel
var ROMountLabel = selinux.ROFileLabel

// SetProcessLabel takes a process label and tells the kernel to assign the
// label to the next program executed by the current process.
// Deprecated: use selinux.SetExecLabel
var SetProcessLabel = selinux.SetExecLabel

// ProcessLabel returns the process label that the kernel will assign
// to the next program executed by the current process.  If "" is returned
// this indicates that the default labeling will happen for the process.
// Deprecated: use selinux.ExecLabel
var ProcessLabel = selinux.ExecLabel

// SetSocketLabel takes a process label and tells the kernel to assign the
// label to the next socket that gets created
// Deprecated: use selinux.SetSocketLabel
var SetSocketLabel = selinux.SetSocketLabel

// SocketLabel retrieves the current default socket label setting
// Deprecated: use selinux.SocketLabel
var SocketLabel = selinux.SocketLabel

// SetKeyLabel takes a process label and tells the kernel to assign the
// label to the next kernel keyring that gets created
// Deprecated: use selinux.SetKeyLabel
var SetKeyLabel = selinux.SetKeyLabel

// KeyLabel retrieves the current default kernel keyring label setting
// Deprecated: use selinux.KeyLabel
var KeyLabel = selinux.KeyLabel

// FileLabel returns the label for specified path
// Deprecated: use selinux.FileLabel
var FileLabel = selinux.FileLabel

// PidLabel will return the label of the process running with the specified pid
// Deprecated: use selinux.PidLabel
var PidLabel = selinux.PidLabel

// Init initialises the labeling system
func Init() {
	_ = selinux.GetEnabled()
}

// ClearLabels will clear all reserved labels
// Deprecated: use selinux.ClearLabels
var ClearLabels = selinux.ClearLabels

// ReserveLabel will record the fact that the MCS label has already been used.
// This will prevent InitLabels from using the MCS label in a newly created
// container
// Deprecated: use selinux.ReserveLabel
func ReserveLabel(label string) error {
	selinux.ReserveLabel(label)
	return nil
}

// ReleaseLabel will remove the reservation of the MCS label.
// This will allow InitLabels to use the MCS label in a newly created
// containers
// Deprecated: use selinux.ReleaseLabel
func ReleaseLabel(label string) error {
	selinux.ReleaseLabel(label)
	return nil
}

// DupSecOpt takes a process label and returns security options that
// can be used to set duplicate labels on future container processes
// Deprecated: use selinux.DupSecOpt
var DupSecOpt = selinux.DupSecOpt

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
