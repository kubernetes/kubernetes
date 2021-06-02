// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package loader

import (
	"fmt"

	"sigs.k8s.io/kustomize/api/filesys"
)

type LoadRestrictorFunc func(
	filesys.FileSystem, filesys.ConfirmedDir, string) (string, error)

func RestrictionRootOnly(
	fSys filesys.FileSystem, root filesys.ConfirmedDir, path string) (string, error) {
	d, f, err := fSys.CleanedAbs(path)
	if err != nil {
		return "", err
	}
	if f == "" {
		return "", fmt.Errorf("'%s' must resolve to a file", path)
	}
	if !d.HasPrefix(root) {
		return "", fmt.Errorf(
			"security; file '%s' is not in or below '%s'",
			path, root)
	}
	return d.Join(f), nil
}

func RestrictionNone(
	_ filesys.FileSystem, _ filesys.ConfirmedDir, path string) (string, error) {
	return path, nil
}
