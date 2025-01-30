/*
Copyright 2023 The Kubernetes Authors.

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
	"io"
	"os"
	"strings"

	"github.com/pkg/errors"

	"k8s.io/klog/v2"
)

// CopyFile copies a file from src to dest.
func CopyFile(src, dest string) error {
	sourceFileInfo, err := os.Stat(src)
	if err != nil {
		return err
	}

	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() {
		_ = sourceFile.Close()
	}()

	destFile, err := os.OpenFile(dest, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, sourceFileInfo.Mode())
	if err != nil {
		return err
	}
	defer func() {
		_ = destFile.Close()
	}()

	_, err = io.Copy(destFile, sourceFile)

	return err
}

// MoveFile moves a file from src to dest.
func MoveFile(src, dest string) error {
	err := os.Rename(src, dest)
	if err != nil && strings.Contains(err.Error(), "invalid cross-device link") {
		// When calling os.Rename(), an "invalid cross-device link" error may occur
		// if the source and destination files are on different file systems.
		// In this case, the file is moved by copying and then deleting the source file,
		// although it is less efficient than os.Rename().
		klog.V(4).Infof("cannot rename %v to %v due to %v, attempting an alternative method", src, dest, err)
		if err := CopyFile(src, dest); err != nil {
			return errors.Wrapf(err, "failed to copy file %v to %v", src, dest)
		}
		return os.Remove(src)
	}
	return err
}
