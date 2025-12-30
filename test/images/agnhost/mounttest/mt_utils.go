//go:build !windows

/*
Copyright 2019 The Kubernetes Authors.

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

package mounttest

import (
	"fmt"
	"os"
	"syscall"
)

func umask(mask int) int {
	return syscall.Umask(mask)
}

func fsType(path string) error {
	if path == "" {
		return nil
	}

	buf := syscall.Statfs_t{}
	if err := syscall.Statfs(path, &buf); err != nil {
		fmt.Printf("error from statfs(%q): %v\n", path, err)
		return err
	}

	fmt.Printf("mount type of %q: %v\n", path, buf.Type)

	return nil
}

func fileMode(path string) error {
	if path == "" {
		return nil
	}

	fileinfo, err := os.Stat(path)
	if err != nil {
		fmt.Printf("error from Stat(%q): %v\n", path, err)
		return err
	}

	fmt.Printf("mode of file %q: %v\n", path, fileinfo.Mode())
	return nil
}

func filePerm(path string) error {
	if path == "" {
		return nil
	}

	fileinfo, err := os.Stat(path)
	if err != nil {
		fmt.Printf("error from Stat(%q): %v\n", path, err)
		return err
	}

	fmt.Printf("perms of file %q: %v\n", path, fileinfo.Mode().Perm())
	return nil
}

func fileOwner(path string) error {
	if path == "" {
		return nil
	}

	buf := syscall.Stat_t{}
	if err := syscall.Stat(path, &buf); err != nil {
		fmt.Printf("error from stat(%q): %v\n", path, err)
		return err
	}

	fmt.Printf("owner UID of %q: %v\n", path, buf.Uid)
	fmt.Printf("owner GID of %q: %v\n", path, buf.Gid)
	return nil
}
