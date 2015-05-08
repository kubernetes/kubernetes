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

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"syscall"
)

var (
	fsTypePath           = ""
	fileModePath         = ""
	readFileContentPath  = ""
	readWriteNewFilePath = ""
)

func init() {
	flag.StringVar(&fsTypePath, "fs_type", "", "Path to print the fs type for")
	flag.StringVar(&fileModePath, "file_mode", "", "Path to print the filemode of")
	flag.StringVar(&readFileContentPath, "file_content", "", "Path to read the file content from")
	flag.StringVar(&readWriteNewFilePath, "rw_new_file", "", "Path to write to and read from")
}

// This program performs some tests on the filesystem as dictated by the
// flags passed by the user.
func main() {
	flag.Parse()

	var (
		err  error
		errs = []error{}
	)

	err = fsType(fsTypePath)
	if err != nil {
		errs = append(errs, err)
	}

	err = readWriteNewFile(readWriteNewFilePath)
	if err != nil {
		errs = append(errs, err)
	}

	err = fileMode(fileModePath)
	if err != nil {
		errs = append(errs, err)
	}

	err = readFileContent(readFileContentPath)
	if err != nil {
		errs = append(errs, err)
	}

	if len(errs) != 0 {
		os.Exit(1)
	}

	os.Exit(0)
}

// Defined by Linux (sys/statfs.h) - the type number for tmpfs mounts.
const linuxTmpfsMagic = 0x01021994

func fsType(path string) error {
	if path == "" {
		return nil
	}

	buf := syscall.Statfs_t{}
	if err := syscall.Statfs(path, &buf); err != nil {
		fmt.Printf("error from statfs(%q): %v", path, err)
		return err
	}

	if buf.Type == linuxTmpfsMagic {
		fmt.Printf("mount type of %q: tmpfs\n", path)
	} else {
		fmt.Printf("mount type of %q: %v\n", path, buf.Type)
	}

	return nil
}

func fileMode(path string) error {
	if path == "" {
		return nil
	}

	fileinfo, err := os.Lstat(path)
	if err != nil {
		fmt.Printf("error from Lstat(%q): %v\n", path, err)
		return err
	}

	fmt.Printf("mode of file %q: %v\n", path, fileinfo.Mode())
	return nil
}

func readFileContent(path string) error {
	if path == "" {
		return nil
	}

	contentBytes, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Printf("error reading file content for %q: %v\n", path, err)
		return err
	}

	fmt.Printf("content of file %q: %v\n", path, string(contentBytes))

	return nil
}

func readWriteNewFile(path string) error {
	if path == "" {
		return nil
	}

	content := "mount-tester new file\n"
	err := ioutil.WriteFile(path, []byte(content), 0644)
	if err != nil {
		fmt.Printf("error writing new file %q: %v\n", path, err)
		return err
	}

	return readFileContent(path)
}
