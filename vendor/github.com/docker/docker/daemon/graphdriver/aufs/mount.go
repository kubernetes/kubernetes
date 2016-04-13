// +build linux

package aufs

import (
	"os/exec"
	"syscall"

	"github.com/Sirupsen/logrus"
)

func Unmount(target string) error {
	if err := exec.Command("auplink", target, "flush").Run(); err != nil {
		logrus.Errorf("Couldn't run auplink before unmount: %s", err)
	}
	if err := syscall.Unmount(target, 0); err != nil {
		return err
	}
	return nil
}
