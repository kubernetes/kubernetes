/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package io

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"

	"github.com/golang/glog"
)

// Writer is an interface which allows to write data to a file.
type Writer interface {
	// WriteFile mimics ioutil.WriteFile.
	WriteFile(filename string, data []byte, perm os.FileMode) error
}

// StdWriter implements Writer interface and uses standard libraries
// for writing data to files.
type StdWriter struct {
}

// WriteFile directly calls ioutil.WriteFile.
func (writer *StdWriter) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return ioutil.WriteFile(filename, data, perm)
}

// NsenterWriter is implementation of Writer interface that allows writing data
// to file using nsenter command.
// If a program (e.g. kubelet) runs in a container it may want to write data to
// a mounted device. Since in Docker, mount propagation mode is set to private,
// it will not see the mounted device in its own namespace. To work around this
// limitation one has to first enter hosts namespace (by using 'nsenter') and
// only then write data.
type NsenterWriter struct{}

// WriteFile calls 'nsenter cat - > <the file>' and 'nsenter chmod' to create a
// file on the host.
func (writer *NsenterWriter) WriteFile(filename string, data []byte, perm os.FileMode) error {
	cmd := "nsenter"
	baseArgs := []string{
		"--mount=/rootfs/proc/1/ns/mnt",
		"--",
	}

	echoArgs := append(baseArgs, "sh", "-c", fmt.Sprintf("cat > %s", filename))
	glog.V(5).Infof("Command to write data to file: %v %v", cmd, echoArgs)
	command := exec.Command(cmd, echoArgs...)
	command.Stdin = bytes.NewBuffer(data)
	outputBytes, err := command.CombinedOutput()
	if err != nil {
		glog.Errorf("Output from writing to %q: %v", filename, string(outputBytes))
		return err
	}

	chmodArgs := append(baseArgs, "chmod", fmt.Sprintf("%o", perm), filename)
	glog.V(5).Infof("Command to change permissions to file: %v %v", cmd, chmodArgs)
	outputBytes, err = exec.Command(cmd, chmodArgs...).CombinedOutput()
	if err != nil {
		glog.Errorf("Output from chmod command: %v", string(outputBytes))
		return err
	}

	return nil
}
