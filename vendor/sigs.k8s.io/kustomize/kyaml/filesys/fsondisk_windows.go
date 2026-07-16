// Copyright 2022 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"path/filepath"

	"golang.org/x/sys/windows"
)

func getOSRoot() (string, error) {
	sysDir, err := windows.GetSystemDirectory()
	if err != nil {
		return "", err
	}
	return filepath.VolumeName(sysDir) + `\`, nil
}
