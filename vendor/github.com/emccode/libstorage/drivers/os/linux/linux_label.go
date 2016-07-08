// +build linux

package linux

import (
	"fmt"
)

/*
formatMountLabel returns a string to be used by the mount command.
The format of this string will be used to alter the labeling of the mountpoint.
The string returned is suitable to be used as the options field of the mount
command.

If you need to have additional mount point options, you can pass them in as
the first parameter.  Second parameter is the label that you wish to apply
to all content in the mount point.
*/
func formatMountLabel(src, mountLabel string) string {
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
