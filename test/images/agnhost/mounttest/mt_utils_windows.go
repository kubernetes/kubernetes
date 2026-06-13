//go:build windows

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
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

func umask(mask int) int {
	// noop for Windows.
	return 0
}

func fileOwner(path string) error {
	// Windows does not have owner UID / GID. However, it has owner SID.
	// not currently implemented in Kubernetes, so noop.
	return nil
}

func fileMode(path string) error {
	if path == "" {
		return nil
	}

	path, err := evalSymlink(path)
	if err != nil {
		return err
	}
	permissions, err := getFilePerm(path)
	if err != nil {
		return err
	}

	fmt.Printf("mode of Windows file %q: %v\n", path, permissions)
	return nil
}

func filePerm(path string) error {
	if path == "" {
		return nil
	}

	path, err := evalSymlink(path)
	if err != nil {
		return err
	}
	permissions, err := getFilePerm(path)
	if err != nil {
		return err
	}

	fmt.Printf("perms of Windows file %q: %v\n", path, permissions)
	return nil
}

func linkFileOwner(path string) error {
	// Windows does not have owner UID / GID. However, it has owner SID.
	// not currently implemented in Kubernetes, so noop.
	return nil
}

func linkFileMode(path string) error {
	if path == "" {
		return nil
	}

	permissions, err := getFilePerm(path)
	if err != nil {
		return err
	}

	fmt.Printf("mode of Windows link file %q: %v\n", path, permissions)
	return nil
}

func linkFilePerm(path string) error {
	if path == "" {
		return nil
	}

	permissions, err := getFilePerm(path)
	if err != nil {
		return err
	}

	fmt.Printf("perms of Windows link file %q: %v\n", path, permissions)
	return nil
}

func evalSymlink(path string) (string, error) {
	info, err := os.Lstat(path)
	if err == nil {
		// go1.23 behavior change: https://github.com/golang/go/issues/63703#issuecomment-2535941458
		if info.Mode()&os.ModeSymlink != 0 || info.Mode()&os.ModeIrregular != 0 {
			evaluated, err := filepath.EvalSymlinks(path)
			if err != nil {
				return "", err
			}
			path = evaluated
		}
	}

	return path, err
}

func getFilePerm(path string) (os.FileMode, error) {
	var (
		out    bytes.Buffer
		errOut bytes.Buffer
	)

	cmd := exec.Command("powershell.exe", "-NonInteractive", "./filePermissions.ps1",
		"-FileName", path)
	cmd.Stdout = &out
	cmd.Stderr = &errOut
	err := cmd.Run()

	if err != nil {
		fmt.Printf("error from PowerShell Script: %v, %v\n", err, errOut.String())
		return 0, err
	}

	output := strings.TrimSpace(out.String())
	val, err := strconv.ParseInt(output, 8, 32)
	if err != nil {
		fmt.Printf("error parsing string '%s' as int: %v\n", output, err)
		return 0, err
	}

	return os.FileMode(val), nil
}

func fsType(path string) error {
	// only NTFS is supported at the moment.
	return nil
}
