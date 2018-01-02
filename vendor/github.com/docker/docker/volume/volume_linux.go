// +build linux

package volume

import (
	"fmt"
	"strings"

	mounttypes "github.com/docker/docker/api/types/mount"
)

// ConvertTmpfsOptions converts *mounttypes.TmpfsOptions to the raw option string
// for mount(2).
func ConvertTmpfsOptions(opt *mounttypes.TmpfsOptions, readOnly bool) (string, error) {
	var rawOpts []string
	if readOnly {
		rawOpts = append(rawOpts, "ro")
	}

	if opt != nil && opt.Mode != 0 {
		rawOpts = append(rawOpts, fmt.Sprintf("mode=%o", opt.Mode))
	}

	if opt != nil && opt.SizeBytes != 0 {
		// calculate suffix here, making this linux specific, but that is
		// okay, since API is that way anyways.

		// we do this by finding the suffix that divides evenly into the
		// value, returning the value itself, with no suffix, if it fails.
		//
		// For the most part, we don't enforce any semantic to this values.
		// The operating system will usually align this and enforce minimum
		// and maximums.
		var (
			size   = opt.SizeBytes
			suffix string
		)
		for _, r := range []struct {
			suffix  string
			divisor int64
		}{
			{"g", 1 << 30},
			{"m", 1 << 20},
			{"k", 1 << 10},
		} {
			if size%r.divisor == 0 {
				size = size / r.divisor
				suffix = r.suffix
				break
			}
		}

		rawOpts = append(rawOpts, fmt.Sprintf("size=%d%s", size, suffix))
	}
	return strings.Join(rawOpts, ","), nil
}
