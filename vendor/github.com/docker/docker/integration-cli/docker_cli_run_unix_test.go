// +build !windows

package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/docker/docker/pkg/mount"
	"github.com/go-check/check"
	"github.com/kr/pty"
)

// #6509
func (s *DockerSuite) TestRunRedirectStdout(c *check.C) {
	checkRedirect := func(command string) {
		_, tty, err := pty.Open()
		if err != nil {
			c.Fatalf("Could not open pty: %v", err)
		}
		cmd := exec.Command("sh", "-c", command)
		cmd.Stdin = tty
		cmd.Stdout = tty
		cmd.Stderr = tty
		if err := cmd.Start(); err != nil {
			c.Fatalf("start err: %v", err)
		}
		ch := make(chan error)
		go func() {
			ch <- cmd.Wait()
			close(ch)
		}()

		select {
		case <-time.After(10 * time.Second):
			c.Fatal("command timeout")
		case err := <-ch:
			if err != nil {
				c.Fatalf("wait err=%v", err)
			}
		}
	}

	checkRedirect(dockerBinary + " run -i busybox cat /etc/passwd | grep -q root")
	checkRedirect(dockerBinary + " run busybox cat /etc/passwd | grep -q root")
}

// Test recursive bind mount works by default
func (s *DockerSuite) TestRunWithVolumesIsRecursive(c *check.C) {
	tmpDir, err := ioutil.TempDir("", "docker_recursive_mount_test")
	if err != nil {
		c.Fatal(err)
	}

	defer os.RemoveAll(tmpDir)

	// Create a temporary tmpfs mount.
	tmpfsDir := filepath.Join(tmpDir, "tmpfs")
	if err := os.MkdirAll(tmpfsDir, 0777); err != nil {
		c.Fatalf("failed to mkdir at %s - %s", tmpfsDir, err)
	}
	if err := mount.Mount("tmpfs", tmpfsDir, "tmpfs", ""); err != nil {
		c.Fatalf("failed to create a tmpfs mount at %s - %s", tmpfsDir, err)
	}

	f, err := ioutil.TempFile(tmpfsDir, "touch-me")
	if err != nil {
		c.Fatal(err)
	}
	defer f.Close()

	runCmd := exec.Command(dockerBinary, "run", "--name", "test-data", "--volume", fmt.Sprintf("%s:/tmp:ro", tmpDir), "busybox:latest", "ls", "/tmp/tmpfs")
	out, stderr, exitCode, err := runCommandWithStdoutStderr(runCmd)
	if err != nil && exitCode != 0 {
		c.Fatal(out, stderr, err)
	}
	if !strings.Contains(out, filepath.Base(f.Name())) {
		c.Fatal("Recursive bind mount test failed. Expected file not found")
	}
}

func (s *DockerSuite) TestRunWithUlimits(c *check.C) {
	testRequires(c, NativeExecDriver)

	out, _ := dockerCmd(c, "run", "--name=testulimits", "--ulimit", "nofile=42", "busybox", "/bin/sh", "-c", "ulimit -n")
	ul := strings.TrimSpace(out)
	if ul != "42" {
		c.Fatalf("expected `ulimit -n` to be 42, got %s", ul)
	}
}

func (s *DockerSuite) TestRunContainerWithCgroupParent(c *check.C) {
	testRequires(c, NativeExecDriver)

	cgroupParent := "test"
	name := "cgroup-test"

	out, _, err := dockerCmdWithError(c, "run", "--cgroup-parent", cgroupParent, "--name", name, "busybox", "cat", "/proc/self/cgroup")
	if err != nil {
		c.Fatalf("unexpected failure when running container with --cgroup-parent option - %s\n%v", string(out), err)
	}
	cgroupPaths := parseCgroupPaths(string(out))
	if len(cgroupPaths) == 0 {
		c.Fatalf("unexpected output - %q", string(out))
	}
	id, err := getIDByName(name)
	c.Assert(err, check.IsNil)
	expectedCgroup := path.Join(cgroupParent, id)
	found := false
	for _, path := range cgroupPaths {
		if strings.HasSuffix(path, expectedCgroup) {
			found = true
			break
		}
	}
	if !found {
		c.Fatalf("unexpected cgroup paths. Expected at least one cgroup path to have suffix %q. Cgroup Paths: %v", expectedCgroup, cgroupPaths)
	}
}

func (s *DockerSuite) TestRunContainerWithCgroupParentAbsPath(c *check.C) {
	testRequires(c, NativeExecDriver)

	cgroupParent := "/cgroup-parent/test"
	name := "cgroup-test"
	out, _, err := dockerCmdWithError(c, "run", "--cgroup-parent", cgroupParent, "--name", name, "busybox", "cat", "/proc/self/cgroup")
	if err != nil {
		c.Fatalf("unexpected failure when running container with --cgroup-parent option - %s\n%v", string(out), err)
	}
	cgroupPaths := parseCgroupPaths(string(out))
	if len(cgroupPaths) == 0 {
		c.Fatalf("unexpected output - %q", string(out))
	}
	id, err := getIDByName(name)
	c.Assert(err, check.IsNil)
	expectedCgroup := path.Join(cgroupParent, id)
	found := false
	for _, path := range cgroupPaths {
		if strings.HasSuffix(path, expectedCgroup) {
			found = true
			break
		}
	}
	if !found {
		c.Fatalf("unexpected cgroup paths. Expected at least one cgroup path to have suffix %q. Cgroup Paths: %v", expectedCgroup, cgroupPaths)
	}
}

func (s *DockerSuite) TestRunContainerWithCgroupMountRO(c *check.C) {
	testRequires(c, NativeExecDriver)

	filename := "/sys/fs/cgroup/devices/test123"
	out, _, err := dockerCmdWithError(c, "run", "busybox", "touch", filename)
	if err == nil {
		c.Fatal("expected cgroup mount point to be read-only, touch file should fail")
	}
	expected := "Read-only file system"
	if !strings.Contains(out, expected) {
		c.Fatalf("expected output from failure to contain %s but contains %s", expected, out)
	}
}

func (s *DockerSuite) TestRunDeviceDirectory(c *check.C) {
	testRequires(c, NativeExecDriver)

	out, _ := dockerCmd(c, "run", "--device", "/dev/snd:/dev/snd", "busybox", "sh", "-c", "ls /dev/snd/")
	if actual := strings.Trim(out, "\r\n"); !strings.Contains(out, "timer") {
		c.Fatalf("expected output /dev/snd/timer, received %s", actual)
	}

	out, _ = dockerCmd(c, "run", "--device", "/dev/snd:/dev/othersnd", "busybox", "sh", "-c", "ls /dev/othersnd/")
	if actual := strings.Trim(out, "\r\n"); !strings.Contains(out, "seq") {
		c.Fatalf("expected output /dev/othersnd/seq, received %s", actual)
	}
}

// TestRunDetach checks attaching and detaching with the escape sequence.
func (s *DockerSuite) TestRunAttachDetach(c *check.C) {
	name := "attach-detach"
	cmd := exec.Command(dockerBinary, "run", "--name", name, "-it", "busybox", "cat")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		c.Fatal(err)
	}
	cpty, tty, err := pty.Open()
	if err != nil {
		c.Fatal(err)
	}
	defer cpty.Close()
	cmd.Stdin = tty
	if err := cmd.Start(); err != nil {
		c.Fatal(err)
	}
	if err := waitRun(name); err != nil {
		c.Fatal(err)
	}

	if _, err := cpty.Write([]byte("hello\n")); err != nil {
		c.Fatal(err)
	}

	out, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil {
		c.Fatal(err)
	}
	if strings.TrimSpace(out) != "hello" {
		c.Fatalf("expected 'hello', got %q", out)
	}

	// escape sequence
	if _, err := cpty.Write([]byte{16}); err != nil {
		c.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := cpty.Write([]byte{17}); err != nil {
		c.Fatal(err)
	}

	ch := make(chan struct{})
	go func() {
		cmd.Wait()
		ch <- struct{}{}
	}()

	running, err := inspectField(name, "State.Running")
	if err != nil {
		c.Fatal(err)
	}
	if running != "true" {
		c.Fatal("expected container to still be running")
	}

	go func() {
		exec.Command(dockerBinary, "kill", name).Run()
	}()

	select {
	case <-ch:
	case <-time.After(10 * time.Millisecond):
		c.Fatal("timed out waiting for container to exit")
	}
}

// "test" should be printed
func (s *DockerSuite) TestRunEchoStdoutWithCPUQuota(c *check.C) {
	testRequires(c, CpuCfsQuota)

	out, _, err := dockerCmdWithError(c, "run", "--cpu-quota", "8000", "--name", "test", "busybox", "echo", "test")
	if err != nil {
		c.Fatalf("failed to run container: %v, output: %q", err, out)
	}
	out = strings.TrimSpace(out)
	if out != "test" {
		c.Errorf("container should've printed 'test'")
	}

	out, err = inspectField("test", "HostConfig.CpuQuota")
	c.Assert(err, check.IsNil)

	if out != "8000" {
		c.Fatalf("setting the CPU CFS quota failed")
	}
}

func (s *DockerSuite) TestRunWithCpuPeriod(c *check.C) {
	testRequires(c, CpuCfsPeriod)

	if _, _, err := dockerCmdWithError(c, "run", "--cpu-period", "50000", "--name", "test", "busybox", "true"); err != nil {
		c.Fatalf("failed to run container: %v", err)
	}

	out, err := inspectField("test", "HostConfig.CpuPeriod")
	c.Assert(err, check.IsNil)
	if out != "50000" {
		c.Fatalf("setting the CPU CFS period failed")
	}
}

func (s *DockerSuite) TestRunOOMExitCode(c *check.C) {
	testRequires(c, OomControl)
	errChan := make(chan error)
	go func() {
		defer close(errChan)
		out, exitCode, _ := dockerCmdWithError(c, "run", "-m", "4MB", "busybox", "sh", "-c", "x=a; while true; do x=$x$x$x$x; done")
		if expected := 137; exitCode != expected {
			errChan <- fmt.Errorf("wrong exit code for OOM container: expected %d, got %d (output: %q)", expected, exitCode, out)
		}
	}()

	select {
	case err := <-errChan:
		c.Assert(err, check.IsNil)
	case <-time.After(30 * time.Second):
		c.Fatal("Timeout waiting for container to die on OOM")
	}
}

func (s *DockerSuite) TestContainerNetworkModeToSelf(c *check.C) {
	out, _, err := dockerCmdWithError(c, "run", "--name=me", "--net=container:me", "busybox", "true")
	if err == nil || !strings.Contains(out, "cannot join own network") {
		c.Fatalf("using container net mode to self should result in an error")
	}
}

func (s *DockerSuite) TestRunContainerNetModeWithDnsMacHosts(c *check.C) {
	out, _, err := dockerCmdWithError(c, "run", "-d", "--name", "parent", "busybox", "top")
	if err != nil {
		c.Fatalf("failed to run container: %v, output: %q", err, out)
	}

	out, _, err = dockerCmdWithError(c, "run", "--dns", "1.2.3.4", "--net=container:parent", "busybox")
	if err == nil || !strings.Contains(out, "Conflicting options: --dns and the network mode") {
		c.Fatalf("run --net=container with --dns should error out")
	}

	out, _, err = dockerCmdWithError(c, "run", "--mac-address", "92:d0:c6:0a:29:33", "--net=container:parent", "busybox")
	if err == nil || !strings.Contains(out, "--mac-address and the network mode") {
		c.Fatalf("run --net=container with --mac-address should error out")
	}

	out, _, err = dockerCmdWithError(c, "run", "--add-host", "test:192.168.2.109", "--net=container:parent", "busybox")
	if err == nil || !strings.Contains(out, "--add-host and the network mode") {
		c.Fatalf("run --net=container with --add-host should error out")
	}
}

func (s *DockerSuite) TestRunContainerNetModeWithExposePort(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "parent", "busybox", "top")

	out, _, err := dockerCmdWithError(c, "run", "-p", "5000:5000", "--net=container:parent", "busybox")
	if err == nil || !strings.Contains(out, "Conflicting options: -p, -P, --publish-all, --publish and the network mode (--net)") {
		c.Fatalf("run --net=container with -p should error out")
	}

	out, _, err = dockerCmdWithError(c, "run", "-P", "--net=container:parent", "busybox")
	if err == nil || !strings.Contains(out, "Conflicting options: -p, -P, --publish-all, --publish and the network mode (--net)") {
		c.Fatalf("run --net=container with -P should error out")
	}

	out, _, err = dockerCmdWithError(c, "run", "--expose", "5000", "--net=container:parent", "busybox")
	if err == nil || !strings.Contains(out, "Conflicting options: --expose and the network mode (--expose)") {
		c.Fatalf("run --net=container with --expose should error out")
	}
}

func (s *DockerSuite) TestRunLinkToContainerNetMode(c *check.C) {
	dockerCmd(c, "run", "--name", "test", "-d", "busybox", "top")
	dockerCmd(c, "run", "--name", "parent", "-d", "--net=container:test", "busybox", "top")
	dockerCmd(c, "run", "-d", "--link=parent:parent", "busybox", "top")
	dockerCmd(c, "run", "--name", "child", "-d", "--net=container:parent", "busybox", "top")
	dockerCmd(c, "run", "-d", "--link=child:child", "busybox", "top")
}

func (s *DockerSuite) TestRunLoopbackOnlyExistsWhenNetworkingDisabled(c *check.C) {
	out, _ := dockerCmd(c, "run", "--net=none", "busybox", "ip", "-o", "-4", "a", "show", "up")

	var (
		count = 0
		parts = strings.Split(out, "\n")
	)

	for _, l := range parts {
		if l != "" {
			count++
		}
	}

	if count != 1 {
		c.Fatalf("Wrong interface count in container %d", count)
	}

	if !strings.HasPrefix(out, "1: lo") {
		c.Fatalf("Wrong interface in test container: expected [1: lo], got %s", out)
	}
}

// Issue #4681
func (s *DockerSuite) TestRunLoopbackWhenNetworkDisabled(c *check.C) {
	dockerCmd(c, "run", "--net=none", "busybox", "ping", "-c", "1", "127.0.0.1")
}

func (s *DockerSuite) TestRunModeNetContainerHostname(c *check.C) {
	testRequires(c, ExecSupport)

	dockerCmd(c, "run", "-i", "-d", "--name", "parent", "busybox", "top")
	out, _ := dockerCmd(c, "exec", "parent", "cat", "/etc/hostname")
	out1, _ := dockerCmd(c, "run", "--net=container:parent", "busybox", "cat", "/etc/hostname")

	if out1 != out {
		c.Fatal("containers with shared net namespace should have same hostname")
	}
}

func (s *DockerSuite) TestRunNetworkNotInitializedNoneMode(c *check.C) {
	out, _, err := dockerCmdWithError(c, "run", "-d", "--net=none", "busybox", "top")
	id := strings.TrimSpace(out)
	res, err := inspectField(id, "NetworkSettings.IPAddress")
	c.Assert(err, check.IsNil)
	if res != "" {
		c.Fatalf("For 'none' mode network must not be initialized, but container got IP: %s", res)
	}
}

func (s *DockerSuite) TestTwoContainersInNetHost(c *check.C) {
	dockerCmd(c, "run", "-d", "--net=host", "--name=first", "busybox", "top")
	dockerCmd(c, "run", "-d", "--net=host", "--name=second", "busybox", "top")
	dockerCmd(c, "stop", "first")
	dockerCmd(c, "stop", "second")
}
