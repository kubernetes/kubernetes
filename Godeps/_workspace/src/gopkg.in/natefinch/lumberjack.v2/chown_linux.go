package lumberjack

import (
	"os"
	"syscall"
)

// os_Chown is a var so we can mock it out during tests.
var os_Chown = os.Chown

func chown(name string, info os.FileInfo) error {
	f, err := os.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode())
	if err != nil {
		return err
	}
	f.Close()
	stat := info.Sys().(*syscall.Stat_t)
	return os_Chown(name, int(stat.Uid), int(stat.Gid))
}
