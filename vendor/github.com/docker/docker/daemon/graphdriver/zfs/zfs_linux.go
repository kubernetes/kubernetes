package zfs

import (
	"fmt"
	"syscall"

	log "github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/graphdriver"
)

func checkRootdirFs(rootdir string) error {
	var buf syscall.Statfs_t
	if err := syscall.Statfs(rootdir, &buf); err != nil {
		return fmt.Errorf("Failed to access '%s': %s", rootdir, err)
	}

	if graphdriver.FsMagic(buf.Type) != graphdriver.FsMagicZfs {
		log.Debugf("[zfs] no zfs dataset found for rootdir '%s'", rootdir)
		return graphdriver.ErrPrerequisites
	}

	return nil
}

func getMountpoint(id string) string {
	return id
}
