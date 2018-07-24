/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"io/ioutil"
	"os"
	"path/filepath"
)

// kubeadm record file path and name define
const (
	kubeadmLogFolder = "/var/lib/kubeadm"
	kubeadmLogFile   = "kubeadm.log"
)

// WriteKubeadmLogFile write record to kubeadm log file when kubeadm init or kubeadm join
func WriteKubeadmLogFile(content string) error {
	return writeKubeadmLogFile(kubeadmLogFolder, kubeadmLogFile, content)
}

func writeKubeadmLogFile(folder, file, content string) error {
	const kubeadmLogPermissions = 0666
	s := []byte(content)
	if _, err := os.Stat(folder); err != nil {
		err := os.MkdirAll(folder, kubeadmLogPermissions)
		if err != nil {
			return err
		}
	}
	if err := ioutil.WriteFile(filepath.Join(folder, file), s, kubeadmLogPermissions); err != nil {
		return err
	}
	return nil
}

// ReadKubeadmLogFile reads the log file from disk and determines if `init` or `join` ran on this machine.
func ReadKubeadmLogFile() (string, error) {
	return readKubeadmLogFile(filepath.Join(kubeadmLogFolder, kubeadmLogFile))
}

func readKubeadmLogFile(filepath string) (string, error) {
	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// RemoveKubeadmLogFile removes the log file and folder when kubeadm reset is called
func RemoveKubeadmLogFile() error {
	return removeKubeadmLogFile(filepath.Join(kubeadmLogFolder, kubeadmLogFile))
}

func removeKubeadmLogFile(filepath string) error {
	return os.Remove(filepath)
}
