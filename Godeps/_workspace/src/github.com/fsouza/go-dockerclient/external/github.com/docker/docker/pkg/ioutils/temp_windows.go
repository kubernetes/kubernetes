// +build windows

package ioutils

import (
	"io/ioutil"

	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/longpath"
)

// TempDir is the equivalent of ioutil.TempDir, except that the result is in Windows longpath format.
func TempDir(dir, prefix string) (string, error) {
	tempDir, err := ioutil.TempDir(dir, prefix)
	if err != nil {
		return "", err
	}
	return longpath.AddPrefix(tempDir), nil
}
