// +build linux,cgo

package loopback

import (
	"fmt"
	"os"

	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

func getLoopbackBackingFile(file *os.File) (uint64, uint64, error) {
	loopInfo, err := ioctlLoopGetStatus64(file.Fd())
	if err != nil {
		logrus.Errorf("Error get loopback backing file: %s", err)
		return 0, 0, ErrGetLoopbackBackingFile
	}
	return loopInfo.loDevice, loopInfo.loInode, nil
}

// SetCapacity reloads the size for the loopback device.
func SetCapacity(file *os.File) error {
	if err := ioctlLoopSetCapacity(file.Fd(), 0); err != nil {
		logrus.Errorf("Error loopbackSetCapacity: %s", err)
		return ErrSetCapacity
	}
	return nil
}

// FindLoopDeviceFor returns a loopback device file for the specified file which
// is backing file of a loop back device.
func FindLoopDeviceFor(file *os.File) *os.File {
	var stat unix.Stat_t
	err := unix.Stat(file.Name(), &stat)
	if err != nil {
		return nil
	}
	targetInode := stat.Ino
	targetDevice := stat.Dev

	for i := 0; true; i++ {
		path := fmt.Sprintf("/dev/loop%d", i)

		file, err := os.OpenFile(path, os.O_RDWR, 0)
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}

			// Ignore all errors until the first not-exist
			// we want to continue looking for the file
			continue
		}

		dev, inode, err := getLoopbackBackingFile(file)
		if err == nil && dev == targetDevice && inode == targetInode {
			return file
		}
		file.Close()
	}

	return nil
}
