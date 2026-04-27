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

package mounttest

import (
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"
)

// CmdMounttest is used by agnhost Cobra.
var CmdMounttest = &cobra.Command{
	Use:   "mounttest",
	Short: "Creates files with given permissions and outputs FS type, owner, mode, permissions, contents of files",
	Long:  "Creates files with specific file permissions, and outputs the filesystem type, owner, mode, permissions, content of the given files, if they exist.",
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

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
	CmdMounttest.Flags().StringVar(&fsTypePath, "fs_type", "", "Path to print the fs type for")
	CmdMounttest.Flags().StringVar(&fileModePath, "file_mode", "", "Path to print the mode bits of")
	CmdMounttest.Flags().StringVar(&filePermPath, "file_perm", "", "Path to print the perms of")
	CmdMounttest.Flags().StringVar(&fileOwnerPath, "file_owner", "", "Path to print the owning UID and GID of")
	CmdMounttest.Flags().StringVar(&newFilePath0644, "new_file_0644", "", "Path to write to and read from with perm 0644")
	CmdMounttest.Flags().StringVar(&newFilePath0666, "new_file_0666", "", "Path to write to and read from with perm 0666")
	CmdMounttest.Flags().StringVar(&newFilePath0660, "new_file_0660", "", "Path to write to and read from with perm 0660")
	CmdMounttest.Flags().StringVar(&newFilePath0777, "new_file_0777", "", "Path to write to and read from with perm 0777")
	CmdMounttest.Flags().StringVar(&readFileContentPath, "file_content", "", "Path to read the file content from")
	CmdMounttest.Flags().StringVar(&readFileContentInLoopPath, "file_content_in_loop", "", "Path to read the file content in loop from")
	CmdMounttest.Flags().IntVar(&retryDuration, "retry_time", 180, "Retry time during the loop")
	CmdMounttest.Flags().BoolVar(&breakOnExpectedContent, "break_on_expected_content", true, "Break out of loop on expected content, (use with --file_content_in_loop flag only)")
}

// This program performs some tests on the filesystem as dictated by the
// flags passed by the user.
func main(cmd *cobra.Command, args []string) {
	var (
		err  error
		errs = []error{}
	)

	// Clear the umask so we can set any mode bits we want.
	umask(0000)

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

func readFileContent(path string) error {
	if path == "" {
		return nil
	}

	contentBytes, err := os.ReadFile(path)
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

	err := os.WriteFile(path, []byte(initialContent), perm)
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
		contentBytes, err = os.ReadFile(filePath)
		if err != nil {
			fmt.Printf("Error reading file %s: %v, retrying\n", filePath, err)
			continue
		}
		fmt.Printf("content of file %q: %v\n", filePath, string(contentBytes))
		if breakOnExpectedContent {
			if string(contentBytes) != initialContent {
				fmt.Printf("Unexpected content. Expected: %s. Retrying\n", initialContent)
				continue
			}
			break
		}
	}
	return err
}
