package operatingsystem

import (
	"bytes"
	"errors"
	"io/ioutil"
)

var (
	// file to use to detect if the daemon is running in a container
	proc1Cgroup = "/proc/1/cgroup"

	// file to check to determine Operating System
	etcOsRelease = "/etc/os-release"
)

func GetOperatingSystem() (string, error) {
	b, err := ioutil.ReadFile(etcOsRelease)
	if err != nil {
		return "", err
	}
	if i := bytes.Index(b, []byte("PRETTY_NAME")); i >= 0 {
		b = b[i+13:]
		return string(b[:bytes.IndexByte(b, '"')]), nil
	}
	return "", errors.New("PRETTY_NAME not found")
}

func IsContainerized() (bool, error) {
	b, err := ioutil.ReadFile(proc1Cgroup)
	if err != nil {
		return false, err
	}
	for _, line := range bytes.Split(b, []byte{'\n'}) {
		if len(line) > 0 && !bytes.HasSuffix(line, []byte{'/'}) {
			return true, nil
		}
	}
	return false, nil
}
