package pidfile

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
)

type PidFile struct {
	path string
}

func checkPidFileAlreadyExists(path string) error {
	if pidString, err := ioutil.ReadFile(path); err == nil {
		if pid, err := strconv.Atoi(string(pidString)); err == nil {
			if _, err := os.Stat(filepath.Join("/proc", string(pid))); err == nil {
				return fmt.Errorf("pid file found, ensure docker is not running or delete %s", path)
			}
		}
	}
	return nil
}

func New(path string) (*PidFile, error) {
	if err := checkPidFileAlreadyExists(path); err != nil {
		return nil, err
	}
	if err := ioutil.WriteFile(path, []byte(fmt.Sprintf("%d", os.Getpid())), 0644); err != nil {
		return nil, err
	}

	return &PidFile{path: path}, nil
}

func (file PidFile) Remove() error {
	if err := os.Remove(file.path); err != nil {
		return err
	}
	return nil
}
