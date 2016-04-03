package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/docker/docker/pkg/archive"
	"github.com/go-check/check"
)

type FileType uint32

const (
	Regular FileType = iota
	Dir
	Symlink
)

type FileData struct {
	filetype FileType
	path     string
	contents string
}

func (fd FileData) creationCommand() string {
	var command string

	switch fd.filetype {
	case Regular:
		// Don't overwrite the file if it already exists!
		command = fmt.Sprintf("if [ ! -f %s ]; then echo %q > %s; fi", fd.path, fd.contents, fd.path)
	case Dir:
		command = fmt.Sprintf("mkdir -p %s", fd.path)
	case Symlink:
		command = fmt.Sprintf("ln -fs %s %s", fd.contents, fd.path)
	}

	return command
}

func mkFilesCommand(fds []FileData) string {
	commands := make([]string, len(fds))

	for i, fd := range fds {
		commands[i] = fd.creationCommand()
	}

	return strings.Join(commands, " && ")
}

var defaultFileData = []FileData{
	{Regular, "file1", "file1"},
	{Regular, "file2", "file2"},
	{Regular, "file3", "file3"},
	{Regular, "file4", "file4"},
	{Regular, "file5", "file5"},
	{Regular, "file6", "file6"},
	{Regular, "file7", "file7"},
	{Dir, "dir1", ""},
	{Regular, "dir1/file1-1", "file1-1"},
	{Regular, "dir1/file1-2", "file1-2"},
	{Dir, "dir2", ""},
	{Regular, "dir2/file2-1", "file2-1"},
	{Regular, "dir2/file2-2", "file2-2"},
	{Dir, "dir3", ""},
	{Regular, "dir3/file3-1", "file3-1"},
	{Regular, "dir3/file3-2", "file3-2"},
	{Dir, "dir4", ""},
	{Regular, "dir4/file3-1", "file4-1"},
	{Regular, "dir4/file3-2", "file4-2"},
	{Dir, "dir5", ""},
	{Symlink, "symlink1", "target1"},
	{Symlink, "symlink2", "target2"},
}

func defaultMkContentCommand() string {
	return mkFilesCommand(defaultFileData)
}

func makeTestContentInDir(c *check.C, dir string) {
	for _, fd := range defaultFileData {
		path := filepath.Join(dir, filepath.FromSlash(fd.path))
		switch fd.filetype {
		case Regular:
			if err := ioutil.WriteFile(path, []byte(fd.contents+"\n"), os.FileMode(0666)); err != nil {
				c.Fatal(err)
			}
		case Dir:
			if err := os.Mkdir(path, os.FileMode(0777)); err != nil {
				c.Fatal(err)
			}
		case Symlink:
			if err := os.Symlink(fd.contents, path); err != nil {
				c.Fatal(err)
			}
		}
	}
}

type testContainerOptions struct {
	addContent bool
	readOnly   bool
	volumes    []string
	workDir    string
	command    string
}

func makeTestContainer(c *check.C, options testContainerOptions) (containerID string) {
	if options.addContent {
		mkContentCmd := defaultMkContentCommand()
		if options.command == "" {
			options.command = mkContentCmd
		} else {
			options.command = fmt.Sprintf("%s && %s", defaultMkContentCommand(), options.command)
		}
	}

	if options.command == "" {
		options.command = "#(nop)"
	}

	args := []string{"run", "-d"}

	for _, volume := range options.volumes {
		args = append(args, "-v", volume)
	}

	if options.workDir != "" {
		args = append(args, "-w", options.workDir)
	}

	if options.readOnly {
		args = append(args, "--read-only")
	}

	args = append(args, "busybox", "/bin/sh", "-c", options.command)

	out, status := dockerCmd(c, args...)
	if status != 0 {
		c.Fatalf("failed to run container, status %d: %s", status, out)
	}

	containerID = strings.TrimSpace(out)

	out, status = dockerCmd(c, "wait", containerID)
	if status != 0 {
		c.Fatalf("failed to wait for test container container, status %d: %s", status, out)
	}

	if exitCode := strings.TrimSpace(out); exitCode != "0" {
		logs, status := dockerCmd(c, "logs", containerID)
		if status != 0 {
			logs = "UNABLE TO GET LOGS"
		}
		c.Fatalf("failed to make test container, exit code (%d): %s", exitCode, logs)
	}

	return
}

func makeCatFileCommand(path string) string {
	return fmt.Sprintf("if [ -f %s ]; then cat %s; fi", path, path)
}

func cpPath(pathElements ...string) string {
	localizedPathElements := make([]string, len(pathElements))
	for i, path := range pathElements {
		localizedPathElements[i] = filepath.FromSlash(path)
	}
	return strings.Join(localizedPathElements, string(filepath.Separator))
}

func cpPathTrailingSep(pathElements ...string) string {
	return fmt.Sprintf("%s%c", cpPath(pathElements...), filepath.Separator)
}

func containerCpPath(containerID string, pathElements ...string) string {
	joined := strings.Join(pathElements, "/")
	return fmt.Sprintf("%s:%s", containerID, joined)
}

func containerCpPathTrailingSep(containerID string, pathElements ...string) string {
	return fmt.Sprintf("%s/", containerCpPath(containerID, pathElements...))
}

func runDockerCp(c *check.C, src, dst string) (err error) {
	c.Logf("running `docker cp %s %s`", src, dst)

	args := []string{"cp", src, dst}

	out, _, err := runCommandWithOutput(exec.Command(dockerBinary, args...))
	if err != nil {
		err = fmt.Errorf("error executing `docker cp` command: %s: %s", err, out)
	}

	return
}

func startContainerGetOutput(c *check.C, cID string) (out string, err error) {
	c.Logf("running `docker start -a %s`", cID)

	args := []string{"start", "-a", cID}

	out, _, err = runCommandWithOutput(exec.Command(dockerBinary, args...))
	if err != nil {
		err = fmt.Errorf("error executing `docker start` command: %s: %s", err, out)
	}

	return
}

func getTestDir(c *check.C, label string) (tmpDir string) {
	var err error

	if tmpDir, err = ioutil.TempDir("", label); err != nil {
		c.Fatalf("unable to make temporary directory: %s", err)
	}

	return
}

func isCpNotExist(err error) bool {
	return strings.Contains(err.Error(), "no such file or directory") || strings.Contains(err.Error(), "cannot find the file specified")
}

func isCpDirNotExist(err error) bool {
	return strings.Contains(err.Error(), archive.ErrDirNotExists.Error())
}

func isCpNotDir(err error) bool {
	return strings.Contains(err.Error(), archive.ErrNotDirectory.Error()) || strings.Contains(err.Error(), "filename, directory name, or volume label syntax is incorrect")
}

func isCpCannotCopyDir(err error) bool {
	return strings.Contains(err.Error(), archive.ErrCannotCopyDir.Error())
}

func isCpCannotCopyReadOnly(err error) bool {
	return strings.Contains(err.Error(), "marked read-only")
}

func isCannotOverwriteNonDirWithDir(err error) bool {
	return strings.Contains(err.Error(), "cannot overwrite non-directory")
}

func fileContentEquals(c *check.C, filename, contents string) (err error) {
	c.Logf("checking that file %q contains %q\n", filename, contents)

	fileBytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return
	}

	expectedBytes, err := ioutil.ReadAll(strings.NewReader(contents))
	if err != nil {
		return
	}

	if !bytes.Equal(fileBytes, expectedBytes) {
		err = fmt.Errorf("file content not equal - expected %q, got %q", string(expectedBytes), string(fileBytes))
	}

	return
}

func containerStartOutputEquals(c *check.C, cID, contents string) (err error) {
	c.Logf("checking that container %q start output contains %q\n", cID, contents)

	out, err := startContainerGetOutput(c, cID)
	if err != nil {
		return err
	}

	if out != contents {
		err = fmt.Errorf("output contents not equal - expected %q, got %q", contents, out)
	}

	return
}

func defaultVolumes(tmpDir string) []string {
	if SameHostDaemon.Condition() {
		return []string{
			"/vol1",
			fmt.Sprintf("%s:/vol2", tmpDir),
			fmt.Sprintf("%s:/vol3", filepath.Join(tmpDir, "vol3")),
			fmt.Sprintf("%s:/vol_ro:ro", filepath.Join(tmpDir, "vol_ro")),
		}
	}

	// Can't bind-mount volumes with separate host daemon.
	return []string{"/vol1", "/vol2", "/vol3", "/vol_ro:/vol_ro:ro"}
}
