// +build linux

package fscommon

import (
	"io/ioutil"
	"os"

	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

func WriteFile(dir, file, data string) error {
	if dir == "" {
		return errors.Errorf("no directory specified for %s", file)
	}
	path, err := securejoin.SecureJoin(dir, file)
	if err != nil {
		return err
	}
	if err := retryingWriteFile(path, []byte(data), 0700); err != nil {
		return errors.Wrapf(err, "failed to write %q to %q", data, path)
	}
	return nil
}

func ReadFile(dir, file string) (string, error) {
	if dir == "" {
		return "", errors.Errorf("no directory specified for %s", file)
	}
	path, err := securejoin.SecureJoin(dir, file)
	if err != nil {
		return "", err
	}
	data, err := ioutil.ReadFile(path)
	return string(data), err
}

func retryingWriteFile(filename string, data []byte, perm os.FileMode) error {
	for {
		err := ioutil.WriteFile(filename, data, perm)
		if errors.Is(err, unix.EINTR) {
			logrus.Infof("interrupted while writing %s to %s", string(data), filename)
			continue
		}
		return err
	}
}
