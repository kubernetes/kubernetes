// +build linux

package devmapper

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// FIXME: this is copy-pasted from the aufs driver.
// It should be moved into the core.

// Mounted returns true if a mount point exists.
func Mounted(mountpoint string) (bool, error) {
	var mntpointSt unix.Stat_t
	if err := unix.Stat(mountpoint, &mntpointSt); err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	var parentSt unix.Stat_t
	if err := unix.Stat(filepath.Join(mountpoint, ".."), &parentSt); err != nil {
		return false, err
	}
	return mntpointSt.Dev != parentSt.Dev, nil
}

type probeData struct {
	fsName string
	magic  string
	offset uint64
}

// ProbeFsType returns the filesystem name for the given device id.
func ProbeFsType(device string) (string, error) {
	probes := []probeData{
		{"btrfs", "_BHRfS_M", 0x10040},
		{"ext4", "\123\357", 0x438},
		{"xfs", "XFSB", 0},
	}

	maxLen := uint64(0)
	for _, p := range probes {
		l := p.offset + uint64(len(p.magic))
		if l > maxLen {
			maxLen = l
		}
	}

	file, err := os.Open(device)
	if err != nil {
		return "", err
	}
	defer file.Close()

	buffer := make([]byte, maxLen)
	l, err := file.Read(buffer)
	if err != nil {
		return "", err
	}

	if uint64(l) != maxLen {
		return "", fmt.Errorf("devmapper: unable to detect filesystem type of %s, short read", device)
	}

	for _, p := range probes {
		if bytes.Equal([]byte(p.magic), buffer[p.offset:p.offset+uint64(len(p.magic))]) {
			return p.fsName, nil
		}
	}

	return "", fmt.Errorf("devmapper: Unknown filesystem type on %s", device)
}

func joinMountOptions(a, b string) string {
	if a == "" {
		return b
	}
	if b == "" {
		return a
	}
	return a + "," + b
}
