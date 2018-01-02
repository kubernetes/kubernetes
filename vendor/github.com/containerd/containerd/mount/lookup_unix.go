// +build !windows

package mount

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	"github.com/pkg/errors"
)

// Lookup returns the mount info corresponds to the path.
func Lookup(dir string) (Info, error) {
	var dirStat syscall.Stat_t
	dir = filepath.Clean(dir)
	if err := syscall.Stat(dir, &dirStat); err != nil {
		return Info{}, errors.Wrapf(err, "failed to access %q", dir)
	}

	mounts, err := Self()
	if err != nil {
		return Info{}, err
	}

	// Sort descending order by Info.Mountpoint
	sort.Slice(mounts, func(i, j int) bool {
		return mounts[j].Mountpoint < mounts[i].Mountpoint
	})
	for _, m := range mounts {
		// Note that m.{Major, Minor} are generally unreliable for our purpose here
		// https://www.spinics.net/lists/linux-btrfs/msg58908.html
		var st syscall.Stat_t
		if err := syscall.Stat(m.Mountpoint, &st); err != nil {
			// may fail; ignore err
			continue
		}
		if st.Dev == dirStat.Dev && strings.HasPrefix(dir, m.Mountpoint) {
			return m, nil
		}
	}

	return Info{}, fmt.Errorf("failed to find the mount info for %q", dir)
}
