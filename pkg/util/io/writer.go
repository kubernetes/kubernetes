/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"

	"github.com/golang/glog"
)

// Writer is an interface which allows to write data to a file.
type Writer interface {
	WriteFile(filename string, data []byte, perm os.FileMode) error
}

// StdWriter implements Writer interface and uses standard libraries
// for writing data to files.
type StdWriter struct {
}

func (writer *StdWriter) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return ioutil.WriteFile(filename, data, perm)
}

// Alternative implementation of Writer interface that allows writing data to file
// using nsenter command.
// If a program (e.g. kubelet) runs in a container it may want to write data to
// a mounted device. Since in Docker, mount propagation mode is set to private,
// it will not see the mounted device in its own namespace. To work around this
// limitaion one has to first enter hosts namespace (by using 'nsenter') and only
// then write data.
type NsenterWriter struct {
}

func (writer *NsenterWriter) WriteFile(filename string, data []byte, perm os.FileMode) error {
	cmd := "nsenter"
	base_args := []string{
		"--mount=/rootfs/proc/1/ns/mnt",
		"--",
	}

	echo_args := append(base_args, "sh", "-c",
		fmt.Sprintf("echo %q | cat > %s", data, filename))
	glog.V(5).Infof("Command to write data to file: %v %v", cmd, echo_args)
	outputBytes, err := exec.Command(cmd, echo_args...).CombinedOutput()
	if err != nil {
		glog.Errorf("Output from writing to %q: %v", filename, string(outputBytes))
		return err
	}

	chmod_args := append(base_args, "chmod", fmt.Sprintf("%o", perm), filename)
	glog.V(5).Infof("Command to change permissions to file: %v %v", cmd, chmod_args)
	outputBytes, err = exec.Command(cmd, chmod_args...).CombinedOutput()
	if err != nil {
		glog.Errorf("Output from chmod command: %v", string(outputBytes))
		return err
	}

	return nil
}
