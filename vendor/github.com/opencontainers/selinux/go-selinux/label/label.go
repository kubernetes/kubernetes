package label

import (
	"fmt"

	"github.com/opencontainers/selinux/go-selinux"
)

// Init initialises the labeling system
func Init() {
	_ = selinux.GetEnabled()
}

// FormatMountLabel returns a string to be used by the mount command. Using
// the SELinux `context` mount option. Changing labels of files on mount
// points with this option can never be changed.
// FormatMountLabel returns a string to be used by the mount command.
// The format of this string will be used to alter the labeling of the mountpoint.
// The string returned is suitable to be used as the options field of the mount command.
// If you need to have additional mount point options, you can pass them in as
// the first parameter.  Second parameter is the label that you wish to apply
// to all content in the mount point.
func FormatMountLabel(src, mountLabel string) string {
	return FormatMountLabelByType(src, mountLabel, "context")
}

// FormatMountLabelByType returns a string to be used by the mount command.
// Allow caller to specify the mount options. For example using the SELinux
// `fscontext` mount option would allow certain container processes to change
// labels of files created on the mount points, where as `context` option does
// not.
// FormatMountLabelByType returns a string to be used by the mount command.
// The format of this string will be used to alter the labeling of the mountpoint.
// The string returned is suitable to be used as the options field of the mount command.
// If you need to have additional mount point options, you can pass them in as
// the first parameter.  Second parameter is the label that you wish to apply
// to all content in the mount point.
func FormatMountLabelByType(src, mountLabel, contextType string) string {
	if mountLabel != "" {
		switch src {
		case "":
			src = fmt.Sprintf("%s=%q", contextType, mountLabel)
		default:
			src = fmt.Sprintf("%s,%s=%q", src, contextType, mountLabel)
		}
	}
	return src
}
