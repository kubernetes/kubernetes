// +build linux

package testutil

import (
	"io/ioutil"
	"os"
	"os/exec"
	"strings"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// NewLoopback creates a loopback device, and returns its device name (/dev/loopX), and its clean-up function.
func NewLoopback(size int64) (string, func() error, error) {
	// create temporary file for the disk image
	file, err := ioutil.TempFile("", "containerd-test-loopback")
	if err != nil {
		return "", nil, errors.Wrap(err, "could not create temporary file for loopback")
	}

	if err := file.Truncate(size); err != nil {
		return "", nil, errors.Wrap(err, "failed to resize temp file")
	}
	file.Close()

	// create device
	losetup := exec.Command("losetup", "--find", "--show", file.Name())
	p, err := losetup.Output()
	if err != nil {
		return "", nil, errors.Wrap(err, "loopback setup failed")
	}

	deviceName := strings.TrimSpace(string(p))
	logrus.Debugf("Created loop device %s (using %s)", deviceName, file.Name())

	cleanup := func() error {
		// detach device
		logrus.Debugf("Removing loop device %s", deviceName)
		losetup := exec.Command("losetup", "--detach", deviceName)
		err := losetup.Run()
		if err != nil {
			return errors.Wrapf(err, "Could not remove loop device %s", deviceName)
		}

		// remove file
		logrus.Debugf("Removing temporary file %s", file.Name())
		return os.Remove(file.Name())
	}

	return deviceName, cleanup, nil
}
