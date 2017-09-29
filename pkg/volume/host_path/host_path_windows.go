/*
Copyright 2017 The Kubernetes Authors.

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

package host_path

import (
	"fmt"
	"os"
	"syscall"

	"k8s.io/api/core/v1"
)

func (dftc *defaultFileTypeChecker) getFileType(info os.FileInfo) (v1.HostPathType, error) {
	mode := info.Sys().(*syscall.Win32FileAttributeData).FileAttributes
	switch mode & syscall.S_IFMT {
	case syscall.S_IFSOCK:
		return v1.HostPathSocket, nil
	case syscall.S_IFBLK:
		return v1.HostPathBlockDev, nil
	case syscall.S_IFCHR:
		return v1.HostPathCharDev, nil
	}
	return "", fmt.Errorf("only recognise socket, block device and character device")
}
