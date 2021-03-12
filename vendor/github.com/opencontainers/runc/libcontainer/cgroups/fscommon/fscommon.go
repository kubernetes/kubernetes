// +build linux

package fscommon

import (
	"bytes"
	"os"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

// WriteFile writes data to a cgroup file in dir.
// It is supposed to be used for cgroup files only.
func WriteFile(dir, file, data string) error {
	fd, err := OpenFile(dir, file, unix.O_WRONLY)
	if err != nil {
		return err
	}
	defer fd.Close()
	if err := retryingWriteFile(fd, data); err != nil {
		return errors.Wrapf(err, "failed to write %q", data)
	}
	return nil
}

// ReadFile reads data from a cgroup file in dir.
// It is supposed to be used for cgroup files only.
func ReadFile(dir, file string) (string, error) {
	fd, err := OpenFile(dir, file, unix.O_RDONLY)
	if err != nil {
		return "", err
	}
	defer fd.Close()
	var buf bytes.Buffer

	_, err = buf.ReadFrom(fd)
	return buf.String(), err
}

func retryingWriteFile(fd *os.File, data string) error {
	for {
		_, err := fd.Write([]byte(data))
		if errors.Is(err, unix.EINTR) {
			logrus.Infof("interrupted while writing %s to %s", data, fd.Name())
			continue
		}
		return err
	}
}
