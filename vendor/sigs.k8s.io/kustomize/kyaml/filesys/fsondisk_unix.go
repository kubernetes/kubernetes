// Copyright 2022 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows
// +build !windows

package filesys

import (
	"path/filepath"
)

func getOSRoot() (string, error) {
	return string(filepath.Separator), nil
}
