// +build linux

package fscommon

import (
	"io/ioutil"

	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/pkg/errors"
)

func WriteFile(dir, file, data string) error {
	if dir == "" {
		return errors.Errorf("no directory specified for %s", file)
	}
	path, err := securejoin.SecureJoin(dir, file)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(path, []byte(data), 0700); err != nil {
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
