// +build darwin dragonfly freebsd linux netbsd openbsd

package godirwalk

import (
	"os"
	"path/filepath"
	"syscall"
)

// modeTypeFromDirent converts a syscall defined constant, which is in purview
// of OS, to a constant defined by Go, assumed by this project to be stable.
//
// When the syscall constant is not recognized, this function falls back to a
// Stat on the file system.
func modeTypeFromDirent(de *syscall.Dirent, osDirname, osBasename string) (os.FileMode, error) {
	switch de.Type {
	case syscall.DT_REG:
		return 0, nil
	case syscall.DT_DIR:
		return os.ModeDir, nil
	case syscall.DT_LNK:
		return os.ModeSymlink, nil
	case syscall.DT_CHR:
		return os.ModeDevice | os.ModeCharDevice, nil
	case syscall.DT_BLK:
		return os.ModeDevice, nil
	case syscall.DT_FIFO:
		return os.ModeNamedPipe, nil
	case syscall.DT_SOCK:
		return os.ModeSocket, nil
	default:
		// If syscall returned unknown type (e.g., DT_UNKNOWN, DT_WHT), then
		// resolve actual mode by reading file information.
		return modeType(filepath.Join(osDirname, osBasename))
	}
}
