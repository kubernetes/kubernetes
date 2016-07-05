/*
Copyright 2015 The Kubernetes Authors.

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
	"time"
)

var (
	fsTypePath                = ""
	fileModePath              = ""
	filePermPath              = ""
	fileOwnerPath             = ""
	newFilePath0644           = ""
	newFilePath0666           = ""
	newFilePath0660           = ""
	newFilePath0777           = ""
	readFileContentPath       = ""
	readFileContentInLoopPath = ""
	retryDuration             = 180
	breakOnExpectedContent    = true
)

func init() {
	flag.StringVar(&fsTypePath, "fs_type", "", "Path to print the fs type for")
	flag.StringVar(&fileModePath, "file_mode", "", "Path to print the mode bits of")
	flag.StringVar(&filePermPath, "file_perm", "", "Path to print the perms of")
	flag.StringVar(&fileOwnerPath, "file_owner", "", "Path to print the owning UID and GID of")
	flag.StringVar(&newFilePath0644, "new_file_0644", "", "Path to write to and read from with perm 0644")
	flag.StringVar(&newFilePath0666, "new_file_0666", "", "Path to write to and read from with perm 0666")
	flag.StringVar(&newFilePath0660, "new_file_0660", "", "Path to write to and read from with perm 0660")
	flag.StringVar(&newFilePath0777, "new_file_0777", "", "Path to write to and read from with perm 0777")
	flag.StringVar(&readFileContentPath, "file_content", "", "Path to read the file content from")
	flag.StringVar(&readFileContentInLoopPath, "file_content_in_loop", "", "Path to read the file content in loop from")
	flag.IntVar(&retryDuration, "retry_time", 180, "Retry time during the loop")
	flag.BoolVar(&breakOnExpectedContent, "break_on_expected_content", true, "Break out of loop on expected content, (use with --file_content_in_loop flag only)")
}

// This program performs some tests on the filesystem as dictated by the
// flags passed by the user.
func main() {
	flag.Parse()

	var (
		err  error
		errs = []error{}
	)

	// Clear the umask so we can set any mode bits we want.
	syscall.Umask(0000)

	// NOTE: the ordering of execution of the various command line
	// flags is intentional and allows a single command to:
	//
	// 1.  Check the fstype of a path
	// 2.  Write a new file within that path
	// 3.  Check that the file's content can be read
	//
	// Changing the ordering of the following code will break tests.

	err = fsType(fsTypePath)
	if err != nil {
		errs = append(errs, err)
	}

	err = readWriteNewFile(newFilePath0644, 0644)
	if err != nil {
		errs = append(errs, err)
	}

	err = readWriteNewFile(newFilePath0666, 0666)
	if err != nil {
		errs = append(errs, err)
	}

	err = readWriteNewFile(newFilePath0660, 0660)
	if err != nil {
		errs = append(errs, err)
	}

	err = readWriteNewFile(newFilePath0777, 0777)
	if err != nil {
		errs = append(errs, err)
	}

	err = fileMode(fileModePath)
	if err != nil {
		errs = append(errs, err)
	}

	err = filePerm(filePermPath)
	if err != nil {
		errs = append(errs, err)
	}

	err = fileOwner(fileOwnerPath)
	if err != nil {
		errs = append(errs, err)
	}

	err = readFileContent(readFileContentPath)
	if err != nil {
		errs = append(errs, err)
	}

	err = readFileContentInLoop(readFileContentInLoopPath, retryDuration, breakOnExpectedContent)
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
		fmt.Printf("error from statfs(%q): %v\n", path, err)
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

const initialContent string = "mount-tester new file\n"

func readWriteNewFile(path string, perm os.FileMode) error {
	if path == "" {
		return nil
	}

	err := ioutil.WriteFile(path, []byte(initialContent), perm)
	if err != nil {
		fmt.Printf("error writing new file %q: %v\n", path, err)
		return err
	}

	return readFileContent(path)
}

func readFileContentInLoop(path string, retryDuration int, breakOnExpectedContent bool) error {
	if path == "" {
		return nil
	}
	return testFileContent(path, retryDuration, breakOnExpectedContent)
}

func testFileContent(filePath string, retryDuration int, breakOnExpectedContent bool) error {
	var (
		contentBytes []byte
		err          error
	)

	retryTime := time.Second * time.Duration(retryDuration)
	for start := time.Now(); time.Since(start) < retryTime; time.Sleep(2 * time.Second) {
		contentBytes, err = ioutil.ReadFile(filePath)
		if err != nil {
			fmt.Printf("Error reading file %s: %v, retrying\n", filePath, err)
			continue
		}
		fmt.Printf("content of file %q: %v\n", filePath, string(contentBytes))
		if breakOnExpectedContent {
			if string(contentBytes) != initialContent {
				fmt.Printf("Unexpected content. Expected: %s. Retrying", initialContent)
				continue
			}
			break
		}
	}
	return err
}
