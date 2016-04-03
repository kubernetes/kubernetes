package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"github.com/go-check/check"
)

const (
	cpTestPathParent = "/some"
	cpTestPath       = "/some/path"
	cpTestName       = "test"
	cpFullPath       = "/some/path/test"

	cpContainerContents = "holla, i am the container"
	cpHostContents      = "hello, i am the host"
)

// Ensure that an all-local path case returns an error.
func (s *DockerSuite) TestCpLocalOnly(c *check.C) {
	err := runDockerCp(c, "foo", "bar")
	if err == nil {
		c.Fatal("expected failure, got success")
	}

	if !strings.Contains(err.Error(), "must specify at least one container source") {
		c.Fatalf("unexpected output: %s", err.Error())
	}
}

// Test for #5656
// Check that garbage paths don't escape the container's rootfs
func (s *DockerSuite) TestCpGarbagePath(c *check.C) {
	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "mkdir -p '"+cpTestPath+"' && echo -n '"+cpContainerContents+"' > "+cpFullPath)
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	if err := os.MkdirAll(cpTestPath, os.ModeDir); err != nil {
		c.Fatal(err)
	}

	hostFile, err := os.Create(cpFullPath)
	if err != nil {
		c.Fatal(err)
	}
	defer hostFile.Close()
	defer os.RemoveAll(cpTestPathParent)

	fmt.Fprintf(hostFile, "%s", cpHostContents)

	tmpdir, err := ioutil.TempDir("", "docker-integration")
	if err != nil {
		c.Fatal(err)
	}

	tmpname := filepath.Join(tmpdir, cpTestName)
	defer os.RemoveAll(tmpdir)

	path := path.Join("../../../../../../../../../../../../", cpFullPath)

	dockerCmd(c, "cp", cleanedContainerID+":"+path, tmpdir)

	file, _ := os.Open(tmpname)
	defer file.Close()

	test, err := ioutil.ReadAll(file)
	if err != nil {
		c.Fatal(err)
	}

	if string(test) == cpHostContents {
		c.Errorf("output matched host file -- garbage path can escape container rootfs")
	}

	if string(test) != cpContainerContents {
		c.Errorf("output doesn't match the input for garbage path")
	}

}

// Check that relative paths are relative to the container's rootfs
func (s *DockerSuite) TestCpRelativePath(c *check.C) {
	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "mkdir -p '"+cpTestPath+"' && echo -n '"+cpContainerContents+"' > "+cpFullPath)
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	if err := os.MkdirAll(cpTestPath, os.ModeDir); err != nil {
		c.Fatal(err)
	}

	hostFile, err := os.Create(cpFullPath)
	if err != nil {
		c.Fatal(err)
	}
	defer hostFile.Close()
	defer os.RemoveAll(cpTestPathParent)

	fmt.Fprintf(hostFile, "%s", cpHostContents)

	tmpdir, err := ioutil.TempDir("", "docker-integration")

	if err != nil {
		c.Fatal(err)
	}

	tmpname := filepath.Join(tmpdir, cpTestName)
	defer os.RemoveAll(tmpdir)

	var relPath string
	if path.IsAbs(cpFullPath) {
		// normally this is `filepath.Rel("/", cpFullPath)` but we cannot
		// get this unix-path manipulation on windows with filepath.
		relPath = cpFullPath[1:]
	} else {
		c.Fatalf("path %s was assumed to be an absolute path", cpFullPath)
	}

	dockerCmd(c, "cp", cleanedContainerID+":"+relPath, tmpdir)

	file, _ := os.Open(tmpname)
	defer file.Close()

	test, err := ioutil.ReadAll(file)
	if err != nil {
		c.Fatal(err)
	}

	if string(test) == cpHostContents {
		c.Errorf("output matched host file -- relative path can escape container rootfs")
	}

	if string(test) != cpContainerContents {
		c.Errorf("output doesn't match the input for relative path")
	}

}

// Check that absolute paths are relative to the container's rootfs
func (s *DockerSuite) TestCpAbsolutePath(c *check.C) {
	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "mkdir -p '"+cpTestPath+"' && echo -n '"+cpContainerContents+"' > "+cpFullPath)
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	if err := os.MkdirAll(cpTestPath, os.ModeDir); err != nil {
		c.Fatal(err)
	}

	hostFile, err := os.Create(cpFullPath)
	if err != nil {
		c.Fatal(err)
	}
	defer hostFile.Close()
	defer os.RemoveAll(cpTestPathParent)

	fmt.Fprintf(hostFile, "%s", cpHostContents)

	tmpdir, err := ioutil.TempDir("", "docker-integration")

	if err != nil {
		c.Fatal(err)
	}

	tmpname := filepath.Join(tmpdir, cpTestName)
	defer os.RemoveAll(tmpdir)

	path := cpFullPath

	dockerCmd(c, "cp", cleanedContainerID+":"+path, tmpdir)

	file, _ := os.Open(tmpname)
	defer file.Close()

	test, err := ioutil.ReadAll(file)
	if err != nil {
		c.Fatal(err)
	}

	if string(test) == cpHostContents {
		c.Errorf("output matched host file -- absolute path can escape container rootfs")
	}

	if string(test) != cpContainerContents {
		c.Errorf("output doesn't match the input for absolute path")
	}

}

// Test for #5619
// Check that absolute symlinks are still relative to the container's rootfs
func (s *DockerSuite) TestCpAbsoluteSymlink(c *check.C) {
	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "mkdir -p '"+cpTestPath+"' && echo -n '"+cpContainerContents+"' > "+cpFullPath+" && ln -s "+cpFullPath+" container_path")
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	if err := os.MkdirAll(cpTestPath, os.ModeDir); err != nil {
		c.Fatal(err)
	}

	hostFile, err := os.Create(cpFullPath)
	if err != nil {
		c.Fatal(err)
	}
	defer hostFile.Close()
	defer os.RemoveAll(cpTestPathParent)

	fmt.Fprintf(hostFile, "%s", cpHostContents)

	tmpdir, err := ioutil.TempDir("", "docker-integration")

	if err != nil {
		c.Fatal(err)
	}

	tmpname := filepath.Join(tmpdir, cpTestName)
	defer os.RemoveAll(tmpdir)

	path := path.Join("/", "container_path")

	dockerCmd(c, "cp", cleanedContainerID+":"+path, tmpdir)

	file, _ := os.Open(tmpname)
	defer file.Close()

	test, err := ioutil.ReadAll(file)
	if err != nil {
		c.Fatal(err)
	}

	if string(test) == cpHostContents {
		c.Errorf("output matched host file -- absolute symlink can escape container rootfs")
	}

	if string(test) != cpContainerContents {
		c.Errorf("output doesn't match the input for absolute symlink")
	}

}

// Test for #5619
// Check that symlinks which are part of the resource path are still relative to the container's rootfs
func (s *DockerSuite) TestCpSymlinkComponent(c *check.C) {
	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "mkdir -p '"+cpTestPath+"' && echo -n '"+cpContainerContents+"' > "+cpFullPath+" && ln -s "+cpTestPath+" container_path")
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	if err := os.MkdirAll(cpTestPath, os.ModeDir); err != nil {
		c.Fatal(err)
	}

	hostFile, err := os.Create(cpFullPath)
	if err != nil {
		c.Fatal(err)
	}
	defer hostFile.Close()
	defer os.RemoveAll(cpTestPathParent)

	fmt.Fprintf(hostFile, "%s", cpHostContents)

	tmpdir, err := ioutil.TempDir("", "docker-integration")

	if err != nil {
		c.Fatal(err)
	}

	tmpname := filepath.Join(tmpdir, cpTestName)
	defer os.RemoveAll(tmpdir)

	path := path.Join("/", "container_path", cpTestName)

	dockerCmd(c, "cp", cleanedContainerID+":"+path, tmpdir)

	file, _ := os.Open(tmpname)
	defer file.Close()

	test, err := ioutil.ReadAll(file)
	if err != nil {
		c.Fatal(err)
	}

	if string(test) == cpHostContents {
		c.Errorf("output matched host file -- symlink path component can escape container rootfs")
	}

	if string(test) != cpContainerContents {
		c.Errorf("output doesn't match the input for symlink path component")
	}

}

// Check that cp with unprivileged user doesn't return any error
func (s *DockerSuite) TestCpUnprivilegedUser(c *check.C) {
	testRequires(c, UnixCli) // uses chmod/su: not available on windows

	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "touch "+cpTestName)
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	tmpdir, err := ioutil.TempDir("", "docker-integration")
	if err != nil {
		c.Fatal(err)
	}

	defer os.RemoveAll(tmpdir)

	if err = os.Chmod(tmpdir, 0777); err != nil {
		c.Fatal(err)
	}

	path := cpTestName

	_, _, err = runCommandWithOutput(exec.Command("su", "unprivilegeduser", "-c", dockerBinary+" cp "+cleanedContainerID+":"+path+" "+tmpdir))
	if err != nil {
		c.Fatalf("couldn't copy with unprivileged user: %s:%s %s", cleanedContainerID, path, err)
	}

}

func (s *DockerSuite) TestCpSpecialFiles(c *check.C) {
	testRequires(c, SameHostDaemon)

	outDir, err := ioutil.TempDir("", "cp-test-special-files")
	if err != nil {
		c.Fatal(err)
	}
	defer os.RemoveAll(outDir)

	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "touch /foo")
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	// Copy actual /etc/resolv.conf
	dockerCmd(c, "cp", cleanedContainerID+":/etc/resolv.conf", outDir)

	expected, err := ioutil.ReadFile("/var/lib/docker/containers/" + cleanedContainerID + "/resolv.conf")
	actual, err := ioutil.ReadFile(outDir + "/resolv.conf")

	if !bytes.Equal(actual, expected) {
		c.Fatalf("Expected copied file to be duplicate of the container resolvconf")
	}

	// Copy actual /etc/hosts
	dockerCmd(c, "cp", cleanedContainerID+":/etc/hosts", outDir)

	expected, err = ioutil.ReadFile("/var/lib/docker/containers/" + cleanedContainerID + "/hosts")
	actual, err = ioutil.ReadFile(outDir + "/hosts")

	if !bytes.Equal(actual, expected) {
		c.Fatalf("Expected copied file to be duplicate of the container hosts")
	}

	// Copy actual /etc/resolv.conf
	dockerCmd(c, "cp", cleanedContainerID+":/etc/hostname", outDir)

	expected, err = ioutil.ReadFile("/var/lib/docker/containers/" + cleanedContainerID + "/hostname")
	actual, err = ioutil.ReadFile(outDir + "/hostname")

	if !bytes.Equal(actual, expected) {
		c.Fatalf("Expected copied file to be duplicate of the container resolvconf")
	}

}

func (s *DockerSuite) TestCpVolumePath(c *check.C) {
	testRequires(c, SameHostDaemon)

	tmpDir, err := ioutil.TempDir("", "cp-test-volumepath")
	if err != nil {
		c.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)
	outDir, err := ioutil.TempDir("", "cp-test-volumepath-out")
	if err != nil {
		c.Fatal(err)
	}
	defer os.RemoveAll(outDir)
	_, err = os.Create(tmpDir + "/test")
	if err != nil {
		c.Fatal(err)
	}

	out, exitCode := dockerCmd(c, "run", "-d", "-v", "/foo", "-v", tmpDir+"/test:/test", "-v", tmpDir+":/baz", "busybox", "/bin/sh", "-c", "touch /foo/bar")
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	// Copy actual volume path
	dockerCmd(c, "cp", cleanedContainerID+":/foo", outDir)

	stat, err := os.Stat(outDir + "/foo")
	if err != nil {
		c.Fatal(err)
	}
	if !stat.IsDir() {
		c.Fatal("expected copied content to be dir")
	}
	stat, err = os.Stat(outDir + "/foo/bar")
	if err != nil {
		c.Fatal(err)
	}
	if stat.IsDir() {
		c.Fatal("Expected file `bar` to be a file")
	}

	// Copy file nested in volume
	dockerCmd(c, "cp", cleanedContainerID+":/foo/bar", outDir)

	stat, err = os.Stat(outDir + "/bar")
	if err != nil {
		c.Fatal(err)
	}
	if stat.IsDir() {
		c.Fatal("Expected file `bar` to be a file")
	}

	// Copy Bind-mounted dir
	dockerCmd(c, "cp", cleanedContainerID+":/baz", outDir)
	stat, err = os.Stat(outDir + "/baz")
	if err != nil {
		c.Fatal(err)
	}
	if !stat.IsDir() {
		c.Fatal("Expected `baz` to be a dir")
	}

	// Copy file nested in bind-mounted dir
	dockerCmd(c, "cp", cleanedContainerID+":/baz/test", outDir)
	fb, err := ioutil.ReadFile(outDir + "/baz/test")
	if err != nil {
		c.Fatal(err)
	}
	fb2, err := ioutil.ReadFile(tmpDir + "/test")
	if err != nil {
		c.Fatal(err)
	}
	if !bytes.Equal(fb, fb2) {
		c.Fatalf("Expected copied file to be duplicate of bind-mounted file")
	}

	// Copy bind-mounted file
	dockerCmd(c, "cp", cleanedContainerID+":/test", outDir)
	fb, err = ioutil.ReadFile(outDir + "/test")
	if err != nil {
		c.Fatal(err)
	}
	fb2, err = ioutil.ReadFile(tmpDir + "/test")
	if err != nil {
		c.Fatal(err)
	}
	if !bytes.Equal(fb, fb2) {
		c.Fatalf("Expected copied file to be duplicate of bind-mounted file")
	}

}

func (s *DockerSuite) TestCpToDot(c *check.C) {
	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "echo lololol > /test")
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	tmpdir, err := ioutil.TempDir("", "docker-integration")
	if err != nil {
		c.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	cwd, err := os.Getwd()
	if err != nil {
		c.Fatal(err)
	}
	defer os.Chdir(cwd)
	if err := os.Chdir(tmpdir); err != nil {
		c.Fatal(err)
	}
	dockerCmd(c, "cp", cleanedContainerID+":/test", ".")
	content, err := ioutil.ReadFile("./test")
	if string(content) != "lololol\n" {
		c.Fatalf("Wrong content in copied file %q, should be %q", content, "lololol\n")
	}
}

func (s *DockerSuite) TestCpToStdout(c *check.C) {
	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "echo lololol > /test")
	if exitCode != 0 {
		c.Fatalf("failed to create a container:%s\n", out)
	}

	cID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cID)
	if strings.TrimSpace(out) != "0" {
		c.Fatalf("failed to set up container:%s\n", out)
	}

	out, _, err := runCommandPipelineWithOutput(
		exec.Command(dockerBinary, "cp", cID+":/test", "-"),
		exec.Command("tar", "-vtf", "-"))

	if err != nil {
		c.Fatalf("Failed to run commands: %s", err)
	}

	if !strings.Contains(out, "test") || !strings.Contains(out, "-rw") {
		c.Fatalf("Missing file from tar TOC:\n%s", out)
	}
}

func (s *DockerSuite) TestCpNameHasColon(c *check.C) {
	testRequires(c, SameHostDaemon)

	out, exitCode := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "echo lololol > /te:s:t")
	if exitCode != 0 {
		c.Fatal("failed to create a container", out)
	}

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "wait", cleanedContainerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

	tmpdir, err := ioutil.TempDir("", "docker-integration")
	if err != nil {
		c.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	dockerCmd(c, "cp", cleanedContainerID+":/te:s:t", tmpdir)
	content, err := ioutil.ReadFile(tmpdir + "/te:s:t")
	if string(content) != "lololol\n" {
		c.Fatalf("Wrong content in copied file %q, should be %q", content, "lololol\n")
	}
}

func (s *DockerSuite) TestCopyAndRestart(c *check.C) {
	expectedMsg := "hello"
	out, _ := dockerCmd(c, "run", "-d", "busybox", "echo", expectedMsg)
	id := strings.TrimSpace(string(out))

	out, _ = dockerCmd(c, "wait", id)

	status := strings.TrimSpace(out)
	if status != "0" {
		c.Fatalf("container exited with status %s", status)
	}

	tmpDir, err := ioutil.TempDir("", "test-docker-restart-after-copy-")
	if err != nil {
		c.Fatalf("unable to make temporary directory: %s", err)
	}
	defer os.RemoveAll(tmpDir)

	dockerCmd(c, "cp", fmt.Sprintf("%s:/etc/issue", id), tmpDir)

	out, _ = dockerCmd(c, "start", "-a", id)

	msg := strings.TrimSpace(out)
	if msg != expectedMsg {
		c.Fatalf("expected %q but got %q", expectedMsg, msg)
	}
}

func (s *DockerSuite) TestCopyCreatedContainer(c *check.C) {
	dockerCmd(c, "create", "--name", "test_cp", "-v", "/test", "busybox")

	tmpDir, err := ioutil.TempDir("", "test")
	if err != nil {
		c.Fatalf("unable to make temporary directory: %s", err)
	}
	defer os.RemoveAll(tmpDir)
	dockerCmd(c, "cp", "test_cp:/bin/sh", tmpDir)
}
