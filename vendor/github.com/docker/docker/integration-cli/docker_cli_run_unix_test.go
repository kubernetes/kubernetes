// +build !windows

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/pkg/homedir"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/sysinfo"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
	"github.com/kr/pty"
)

// #6509
func (s *DockerSuite) TestRunRedirectStdout(c *check.C) {
	checkRedirect := func(command string) {
		_, tty, err := pty.Open()
		c.Assert(err, checker.IsNil, check.Commentf("Could not open pty"))
		cmd := exec.Command("sh", "-c", command)
		cmd.Stdin = tty
		cmd.Stdout = tty
		cmd.Stderr = tty
		c.Assert(cmd.Start(), checker.IsNil)
		ch := make(chan error)
		go func() {
			ch <- cmd.Wait()
			close(ch)
		}()

		select {
		case <-time.After(10 * time.Second):
			c.Fatal("command timeout")
		case err := <-ch:
			c.Assert(err, checker.IsNil, check.Commentf("wait err"))
		}
	}

	checkRedirect(dockerBinary + " run -i busybox cat /etc/passwd | grep -q root")
	checkRedirect(dockerBinary + " run busybox cat /etc/passwd | grep -q root")
}

// Test recursive bind mount works by default
func (s *DockerSuite) TestRunWithVolumesIsRecursive(c *check.C) {
	// /tmp gets permission denied
	testRequires(c, NotUserNamespace, SameHostDaemon)
	tmpDir, err := ioutil.TempDir("", "docker_recursive_mount_test")
	c.Assert(err, checker.IsNil)

	defer os.RemoveAll(tmpDir)

	// Create a temporary tmpfs mount.
	tmpfsDir := filepath.Join(tmpDir, "tmpfs")
	c.Assert(os.MkdirAll(tmpfsDir, 0777), checker.IsNil, check.Commentf("failed to mkdir at %s", tmpfsDir))
	c.Assert(mount.Mount("tmpfs", tmpfsDir, "tmpfs", ""), checker.IsNil, check.Commentf("failed to create a tmpfs mount at %s", tmpfsDir))

	f, err := ioutil.TempFile(tmpfsDir, "touch-me")
	c.Assert(err, checker.IsNil)
	defer f.Close()

	out, _ := dockerCmd(c, "run", "--name", "test-data", "--volume", fmt.Sprintf("%s:/tmp:ro", tmpDir), "busybox:latest", "ls", "/tmp/tmpfs")
	c.Assert(out, checker.Contains, filepath.Base(f.Name()), check.Commentf("Recursive bind mount test failed. Expected file not found"))
}

func (s *DockerSuite) TestRunDeviceDirectory(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace, NotArm)
	if _, err := os.Stat("/dev/snd"); err != nil {
		c.Skip("Host does not have /dev/snd")
	}

	out, _ := dockerCmd(c, "run", "--device", "/dev/snd:/dev/snd", "busybox", "sh", "-c", "ls /dev/snd/")
	c.Assert(strings.Trim(out, "\r\n"), checker.Contains, "timer", check.Commentf("expected output /dev/snd/timer"))

	out, _ = dockerCmd(c, "run", "--device", "/dev/snd:/dev/othersnd", "busybox", "sh", "-c", "ls /dev/othersnd/")
	c.Assert(strings.Trim(out, "\r\n"), checker.Contains, "seq", check.Commentf("expected output /dev/othersnd/seq"))
}

// TestRunAttachDetach checks attaching and detaching with the default escape sequence.
func (s *DockerSuite) TestRunAttachDetach(c *check.C) {
	name := "attach-detach"

	dockerCmd(c, "run", "--name", name, "-itd", "busybox", "cat")

	cmd := exec.Command(dockerBinary, "attach", name)
	stdout, err := cmd.StdoutPipe()
	c.Assert(err, checker.IsNil)
	cpty, tty, err := pty.Open()
	c.Assert(err, checker.IsNil)
	defer cpty.Close()
	cmd.Stdin = tty
	c.Assert(cmd.Start(), checker.IsNil)
	c.Assert(waitRun(name), check.IsNil)

	_, err = cpty.Write([]byte("hello\n"))
	c.Assert(err, checker.IsNil)

	out, err := bufio.NewReader(stdout).ReadString('\n')
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello")

	// escape sequence
	_, err = cpty.Write([]byte{16})
	c.Assert(err, checker.IsNil)
	time.Sleep(100 * time.Millisecond)
	_, err = cpty.Write([]byte{17})
	c.Assert(err, checker.IsNil)

	ch := make(chan struct{})
	go func() {
		cmd.Wait()
		ch <- struct{}{}
	}()

	select {
	case <-ch:
	case <-time.After(10 * time.Second):
		c.Fatal("timed out waiting for container to exit")
	}

	running := inspectField(c, name, "State.Running")
	c.Assert(running, checker.Equals, "true", check.Commentf("expected container to still be running"))

	out, _ = dockerCmd(c, "events", "--since=0", "--until", daemonUnixTime(c), "-f", "container="+name)
	// attach and detach event should be monitored
	c.Assert(out, checker.Contains, "attach")
	c.Assert(out, checker.Contains, "detach")
}

// TestRunAttachDetachFromFlag checks attaching and detaching with the escape sequence specified via flags.
func (s *DockerSuite) TestRunAttachDetachFromFlag(c *check.C) {
	name := "attach-detach"
	keyCtrlA := []byte{1}
	keyA := []byte{97}

	dockerCmd(c, "run", "--name", name, "-itd", "busybox", "cat")

	cmd := exec.Command(dockerBinary, "attach", "--detach-keys=ctrl-a,a", name)
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
	c.Assert(waitRun(name), check.IsNil)

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
	if _, err := cpty.Write(keyCtrlA); err != nil {
		c.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := cpty.Write(keyA); err != nil {
		c.Fatal(err)
	}

	ch := make(chan struct{})
	go func() {
		cmd.Wait()
		ch <- struct{}{}
	}()

	select {
	case <-ch:
	case <-time.After(10 * time.Second):
		c.Fatal("timed out waiting for container to exit")
	}

	running := inspectField(c, name, "State.Running")
	c.Assert(running, checker.Equals, "true", check.Commentf("expected container to still be running"))
}

// TestRunAttachDetachFromInvalidFlag checks attaching and detaching with the escape sequence specified via flags.
func (s *DockerSuite) TestRunAttachDetachFromInvalidFlag(c *check.C) {
	name := "attach-detach"
	dockerCmd(c, "run", "--name", name, "-itd", "busybox", "top")
	c.Assert(waitRun(name), check.IsNil)

	// specify an invalid detach key, container will ignore it and use default
	cmd := exec.Command(dockerBinary, "attach", "--detach-keys=ctrl-A,a", name)
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

	bufReader := bufio.NewReader(stdout)
	out, err := bufReader.ReadString('\n')
	if err != nil {
		c.Fatal(err)
	}
	// it should print a warning to indicate the detach key flag is invalid
	errStr := "Invalid detach keys (ctrl-A,a) provided"
	c.Assert(strings.TrimSpace(out), checker.Equals, errStr)
}

// TestRunAttachDetachFromConfig checks attaching and detaching with the escape sequence specified via config file.
func (s *DockerSuite) TestRunAttachDetachFromConfig(c *check.C) {
	keyCtrlA := []byte{1}
	keyA := []byte{97}

	// Setup config
	homeKey := homedir.Key()
	homeVal := homedir.Get()
	tmpDir, err := ioutil.TempDir("", "fake-home")
	c.Assert(err, checker.IsNil)
	defer os.RemoveAll(tmpDir)

	dotDocker := filepath.Join(tmpDir, ".docker")
	os.Mkdir(dotDocker, 0600)
	tmpCfg := filepath.Join(dotDocker, "config.json")

	defer func() { os.Setenv(homeKey, homeVal) }()
	os.Setenv(homeKey, tmpDir)

	data := `{
		"detachKeys": "ctrl-a,a"
	}`

	err = ioutil.WriteFile(tmpCfg, []byte(data), 0600)
	c.Assert(err, checker.IsNil)

	// Then do the work
	name := "attach-detach"
	dockerCmd(c, "run", "--name", name, "-itd", "busybox", "cat")

	cmd := exec.Command(dockerBinary, "attach", name)
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
	c.Assert(waitRun(name), check.IsNil)

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
	if _, err := cpty.Write(keyCtrlA); err != nil {
		c.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := cpty.Write(keyA); err != nil {
		c.Fatal(err)
	}

	ch := make(chan struct{})
	go func() {
		cmd.Wait()
		ch <- struct{}{}
	}()

	select {
	case <-ch:
	case <-time.After(10 * time.Second):
		c.Fatal("timed out waiting for container to exit")
	}

	running := inspectField(c, name, "State.Running")
	c.Assert(running, checker.Equals, "true", check.Commentf("expected container to still be running"))
}

// TestRunAttachDetachKeysOverrideConfig checks attaching and detaching with the detach flags, making sure it overrides config file
func (s *DockerSuite) TestRunAttachDetachKeysOverrideConfig(c *check.C) {
	keyCtrlA := []byte{1}
	keyA := []byte{97}

	// Setup config
	homeKey := homedir.Key()
	homeVal := homedir.Get()
	tmpDir, err := ioutil.TempDir("", "fake-home")
	c.Assert(err, checker.IsNil)
	defer os.RemoveAll(tmpDir)

	dotDocker := filepath.Join(tmpDir, ".docker")
	os.Mkdir(dotDocker, 0600)
	tmpCfg := filepath.Join(dotDocker, "config.json")

	defer func() { os.Setenv(homeKey, homeVal) }()
	os.Setenv(homeKey, tmpDir)

	data := `{
		"detachKeys": "ctrl-e,e"
	}`

	err = ioutil.WriteFile(tmpCfg, []byte(data), 0600)
	c.Assert(err, checker.IsNil)

	// Then do the work
	name := "attach-detach"
	dockerCmd(c, "run", "--name", name, "-itd", "busybox", "cat")

	cmd := exec.Command(dockerBinary, "attach", "--detach-keys=ctrl-a,a", name)
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
	c.Assert(waitRun(name), check.IsNil)

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
	if _, err := cpty.Write(keyCtrlA); err != nil {
		c.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := cpty.Write(keyA); err != nil {
		c.Fatal(err)
	}

	ch := make(chan struct{})
	go func() {
		cmd.Wait()
		ch <- struct{}{}
	}()

	select {
	case <-ch:
	case <-time.After(10 * time.Second):
		c.Fatal("timed out waiting for container to exit")
	}

	running := inspectField(c, name, "State.Running")
	c.Assert(running, checker.Equals, "true", check.Commentf("expected container to still be running"))
}

func (s *DockerSuite) TestRunAttachInvalidDetachKeySequencePreserved(c *check.C) {
	name := "attach-detach"
	keyA := []byte{97}
	keyB := []byte{98}

	dockerCmd(c, "run", "--name", name, "-itd", "busybox", "cat")

	cmd := exec.Command(dockerBinary, "attach", "--detach-keys=a,b,c", name)
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
	c.Assert(waitRun(name), check.IsNil)

	// Invalid escape sequence aba, should print aba in output
	if _, err := cpty.Write(keyA); err != nil {
		c.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := cpty.Write(keyB); err != nil {
		c.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := cpty.Write(keyA); err != nil {
		c.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := cpty.Write([]byte("\n")); err != nil {
		c.Fatal(err)
	}

	out, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil {
		c.Fatal(err)
	}
	if strings.TrimSpace(out) != "aba" {
		c.Fatalf("expected 'aba', got %q", out)
	}
}

// "test" should be printed
func (s *DockerSuite) TestRunWithCPUQuota(c *check.C) {
	testRequires(c, cpuCfsQuota)

	file := "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
	out, _ := dockerCmd(c, "run", "--cpu-quota", "8000", "--name", "test", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "8000")

	out = inspectField(c, "test", "HostConfig.CpuQuota")
	c.Assert(out, checker.Equals, "8000", check.Commentf("setting the CPU CFS quota failed"))
}

func (s *DockerSuite) TestRunWithCpuPeriod(c *check.C) {
	testRequires(c, cpuCfsPeriod)

	file := "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
	out, _ := dockerCmd(c, "run", "--cpu-period", "50000", "--name", "test", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "50000")

	out, _ = dockerCmd(c, "run", "--cpu-period", "0", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "100000")

	out = inspectField(c, "test", "HostConfig.CpuPeriod")
	c.Assert(out, checker.Equals, "50000", check.Commentf("setting the CPU CFS period failed"))
}

func (s *DockerSuite) TestRunWithInvalidCpuPeriod(c *check.C) {
	testRequires(c, cpuCfsPeriod)
	out, _, err := dockerCmdWithError("run", "--cpu-period", "900", "busybox", "true")
	c.Assert(err, check.NotNil)
	expected := "CPU cfs period can not be less than 1ms (i.e. 1000) or larger than 1s (i.e. 1000000)"
	c.Assert(out, checker.Contains, expected)

	out, _, err = dockerCmdWithError("run", "--cpu-period", "2000000", "busybox", "true")
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, expected)

	out, _, err = dockerCmdWithError("run", "--cpu-period", "-3", "busybox", "true")
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, expected)
}

func (s *DockerSuite) TestRunWithKernelMemory(c *check.C) {
	testRequires(c, kernelMemorySupport)

	file := "/sys/fs/cgroup/memory/memory.kmem.limit_in_bytes"
	cli.DockerCmd(c, "run", "--kernel-memory", "50M", "--name", "test1", "busybox", "cat", file).Assert(c, icmd.Expected{
		Out: "52428800",
	})

	cli.InspectCmd(c, "test1", cli.Format(".HostConfig.KernelMemory")).Assert(c, icmd.Expected{
		Out: "52428800",
	})
}

func (s *DockerSuite) TestRunWithInvalidKernelMemory(c *check.C) {
	testRequires(c, kernelMemorySupport)

	out, _, err := dockerCmdWithError("run", "--kernel-memory", "2M", "busybox", "true")
	c.Assert(err, check.NotNil)
	expected := "Minimum kernel memory limit allowed is 4MB"
	c.Assert(out, checker.Contains, expected)

	out, _, err = dockerCmdWithError("run", "--kernel-memory", "-16m", "--name", "test2", "busybox", "echo", "test")
	c.Assert(err, check.NotNil)
	expected = "invalid size"
	c.Assert(out, checker.Contains, expected)
}

func (s *DockerSuite) TestRunWithCPUShares(c *check.C) {
	testRequires(c, cpuShare)

	file := "/sys/fs/cgroup/cpu/cpu.shares"
	out, _ := dockerCmd(c, "run", "--cpu-shares", "1000", "--name", "test", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "1000")

	out = inspectField(c, "test", "HostConfig.CPUShares")
	c.Assert(out, check.Equals, "1000")
}

// "test" should be printed
func (s *DockerSuite) TestRunEchoStdoutWithCPUSharesAndMemoryLimit(c *check.C) {
	testRequires(c, cpuShare)
	testRequires(c, memoryLimitSupport)
	cli.DockerCmd(c, "run", "--cpu-shares", "1000", "-m", "32m", "busybox", "echo", "test").Assert(c, icmd.Expected{
		Out: "test\n",
	})
}

func (s *DockerSuite) TestRunWithCpusetCpus(c *check.C) {
	testRequires(c, cgroupCpuset)

	file := "/sys/fs/cgroup/cpuset/cpuset.cpus"
	out, _ := dockerCmd(c, "run", "--cpuset-cpus", "0", "--name", "test", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "0")

	out = inspectField(c, "test", "HostConfig.CpusetCpus")
	c.Assert(out, check.Equals, "0")
}

func (s *DockerSuite) TestRunWithCpusetMems(c *check.C) {
	testRequires(c, cgroupCpuset)

	file := "/sys/fs/cgroup/cpuset/cpuset.mems"
	out, _ := dockerCmd(c, "run", "--cpuset-mems", "0", "--name", "test", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "0")

	out = inspectField(c, "test", "HostConfig.CpusetMems")
	c.Assert(out, check.Equals, "0")
}

func (s *DockerSuite) TestRunWithBlkioWeight(c *check.C) {
	testRequires(c, blkioWeight)

	file := "/sys/fs/cgroup/blkio/blkio.weight"
	out, _ := dockerCmd(c, "run", "--blkio-weight", "300", "--name", "test", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "300")

	out = inspectField(c, "test", "HostConfig.BlkioWeight")
	c.Assert(out, check.Equals, "300")
}

func (s *DockerSuite) TestRunWithInvalidBlkioWeight(c *check.C) {
	testRequires(c, blkioWeight)
	out, _, err := dockerCmdWithError("run", "--blkio-weight", "5", "busybox", "true")
	c.Assert(err, check.NotNil, check.Commentf(out))
	expected := "Range of blkio weight is from 10 to 1000"
	c.Assert(out, checker.Contains, expected)
}

func (s *DockerSuite) TestRunWithInvalidPathforBlkioWeightDevice(c *check.C) {
	testRequires(c, blkioWeight)
	out, _, err := dockerCmdWithError("run", "--blkio-weight-device", "/dev/sdX:100", "busybox", "true")
	c.Assert(err, check.NotNil, check.Commentf(out))
}

func (s *DockerSuite) TestRunWithInvalidPathforBlkioDeviceReadBps(c *check.C) {
	testRequires(c, blkioWeight)
	out, _, err := dockerCmdWithError("run", "--device-read-bps", "/dev/sdX:500", "busybox", "true")
	c.Assert(err, check.NotNil, check.Commentf(out))
}

func (s *DockerSuite) TestRunWithInvalidPathforBlkioDeviceWriteBps(c *check.C) {
	testRequires(c, blkioWeight)
	out, _, err := dockerCmdWithError("run", "--device-write-bps", "/dev/sdX:500", "busybox", "true")
	c.Assert(err, check.NotNil, check.Commentf(out))
}

func (s *DockerSuite) TestRunWithInvalidPathforBlkioDeviceReadIOps(c *check.C) {
	testRequires(c, blkioWeight)
	out, _, err := dockerCmdWithError("run", "--device-read-iops", "/dev/sdX:500", "busybox", "true")
	c.Assert(err, check.NotNil, check.Commentf(out))
}

func (s *DockerSuite) TestRunWithInvalidPathforBlkioDeviceWriteIOps(c *check.C) {
	testRequires(c, blkioWeight)
	out, _, err := dockerCmdWithError("run", "--device-write-iops", "/dev/sdX:500", "busybox", "true")
	c.Assert(err, check.NotNil, check.Commentf(out))
}

func (s *DockerSuite) TestRunOOMExitCode(c *check.C) {
	testRequires(c, memoryLimitSupport, swapMemorySupport)
	errChan := make(chan error)
	go func() {
		defer close(errChan)
		out, exitCode, _ := dockerCmdWithError("run", "-m", "4MB", "busybox", "sh", "-c", "x=a; while true; do x=$x$x$x$x; done")
		if expected := 137; exitCode != expected {
			errChan <- fmt.Errorf("wrong exit code for OOM container: expected %d, got %d (output: %q)", expected, exitCode, out)
		}
	}()

	select {
	case err := <-errChan:
		c.Assert(err, check.IsNil)
	case <-time.After(600 * time.Second):
		c.Fatal("Timeout waiting for container to die on OOM")
	}
}

func (s *DockerSuite) TestRunWithMemoryLimit(c *check.C) {
	testRequires(c, memoryLimitSupport)

	file := "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	cli.DockerCmd(c, "run", "-m", "32M", "--name", "test", "busybox", "cat", file).Assert(c, icmd.Expected{
		Out: "33554432",
	})
	cli.InspectCmd(c, "test", cli.Format(".HostConfig.Memory")).Assert(c, icmd.Expected{
		Out: "33554432",
	})
}

// TestRunWithoutMemoryswapLimit sets memory limit and disables swap
// memory limit, this means the processes in the container can use
// 16M memory and as much swap memory as they need (if the host
// supports swap memory).
func (s *DockerSuite) TestRunWithoutMemoryswapLimit(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)
	testRequires(c, swapMemorySupport)
	dockerCmd(c, "run", "-m", "32m", "--memory-swap", "-1", "busybox", "true")
}

func (s *DockerSuite) TestRunWithSwappiness(c *check.C) {
	testRequires(c, memorySwappinessSupport)
	file := "/sys/fs/cgroup/memory/memory.swappiness"
	out, _ := dockerCmd(c, "run", "--memory-swappiness", "0", "--name", "test", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "0")

	out = inspectField(c, "test", "HostConfig.MemorySwappiness")
	c.Assert(out, check.Equals, "0")
}

func (s *DockerSuite) TestRunWithSwappinessInvalid(c *check.C) {
	testRequires(c, memorySwappinessSupport)
	out, _, err := dockerCmdWithError("run", "--memory-swappiness", "101", "busybox", "true")
	c.Assert(err, check.NotNil)
	expected := "Valid memory swappiness range is 0-100"
	c.Assert(out, checker.Contains, expected, check.Commentf("Expected output to contain %q, not %q", out, expected))

	out, _, err = dockerCmdWithError("run", "--memory-swappiness", "-10", "busybox", "true")
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, expected, check.Commentf("Expected output to contain %q, not %q", out, expected))
}

func (s *DockerSuite) TestRunWithMemoryReservation(c *check.C) {
	testRequires(c, memoryReservationSupport)

	file := "/sys/fs/cgroup/memory/memory.soft_limit_in_bytes"
	out, _ := dockerCmd(c, "run", "--memory-reservation", "200M", "--name", "test", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "209715200")

	out = inspectField(c, "test", "HostConfig.MemoryReservation")
	c.Assert(out, check.Equals, "209715200")
}

func (s *DockerSuite) TestRunWithMemoryReservationInvalid(c *check.C) {
	testRequires(c, memoryLimitSupport)
	testRequires(c, memoryReservationSupport)
	out, _, err := dockerCmdWithError("run", "-m", "500M", "--memory-reservation", "800M", "busybox", "true")
	c.Assert(err, check.NotNil)
	expected := "Minimum memory limit can not be less than memory reservation limit"
	c.Assert(strings.TrimSpace(out), checker.Contains, expected, check.Commentf("run container should fail with invalid memory reservation"))

	out, _, err = dockerCmdWithError("run", "--memory-reservation", "1k", "busybox", "true")
	c.Assert(err, check.NotNil)
	expected = "Minimum memory reservation allowed is 4MB"
	c.Assert(strings.TrimSpace(out), checker.Contains, expected, check.Commentf("run container should fail with invalid memory reservation"))
}

func (s *DockerSuite) TestStopContainerSignal(c *check.C) {
	out, _ := dockerCmd(c, "run", "--stop-signal", "SIGUSR1", "-d", "busybox", "/bin/sh", "-c", `trap 'echo "exit trapped"; exit 0' USR1; while true; do sleep 1; done`)
	containerID := strings.TrimSpace(out)

	c.Assert(waitRun(containerID), checker.IsNil)

	dockerCmd(c, "stop", containerID)
	out, _ = dockerCmd(c, "logs", containerID)

	c.Assert(out, checker.Contains, "exit trapped", check.Commentf("Expected `exit trapped` in the log"))
}

func (s *DockerSuite) TestRunSwapLessThanMemoryLimit(c *check.C) {
	testRequires(c, memoryLimitSupport)
	testRequires(c, swapMemorySupport)
	out, _, err := dockerCmdWithError("run", "-m", "16m", "--memory-swap", "15m", "busybox", "echo", "test")
	expected := "Minimum memoryswap limit should be larger than memory limit"
	c.Assert(err, check.NotNil)

	c.Assert(out, checker.Contains, expected)
}

func (s *DockerSuite) TestRunInvalidCpusetCpusFlagValue(c *check.C) {
	testRequires(c, cgroupCpuset, SameHostDaemon)

	sysInfo := sysinfo.New(true)
	cpus, err := parsers.ParseUintList(sysInfo.Cpus)
	c.Assert(err, check.IsNil)
	var invalid int
	for i := 0; i <= len(cpus)+1; i++ {
		if !cpus[i] {
			invalid = i
			break
		}
	}
	out, _, err := dockerCmdWithError("run", "--cpuset-cpus", strconv.Itoa(invalid), "busybox", "true")
	c.Assert(err, check.NotNil)
	expected := fmt.Sprintf("Error response from daemon: Requested CPUs are not available - requested %s, available: %s", strconv.Itoa(invalid), sysInfo.Cpus)
	c.Assert(out, checker.Contains, expected)
}

func (s *DockerSuite) TestRunInvalidCpusetMemsFlagValue(c *check.C) {
	testRequires(c, cgroupCpuset)

	sysInfo := sysinfo.New(true)
	mems, err := parsers.ParseUintList(sysInfo.Mems)
	c.Assert(err, check.IsNil)
	var invalid int
	for i := 0; i <= len(mems)+1; i++ {
		if !mems[i] {
			invalid = i
			break
		}
	}
	out, _, err := dockerCmdWithError("run", "--cpuset-mems", strconv.Itoa(invalid), "busybox", "true")
	c.Assert(err, check.NotNil)
	expected := fmt.Sprintf("Error response from daemon: Requested memory nodes are not available - requested %s, available: %s", strconv.Itoa(invalid), sysInfo.Mems)
	c.Assert(out, checker.Contains, expected)
}

func (s *DockerSuite) TestRunInvalidCPUShares(c *check.C) {
	testRequires(c, cpuShare, DaemonIsLinux)
	out, _, err := dockerCmdWithError("run", "--cpu-shares", "1", "busybox", "echo", "test")
	c.Assert(err, check.NotNil, check.Commentf(out))
	expected := "The minimum allowed cpu-shares is 2"
	c.Assert(out, checker.Contains, expected)

	out, _, err = dockerCmdWithError("run", "--cpu-shares", "-1", "busybox", "echo", "test")
	c.Assert(err, check.NotNil, check.Commentf(out))
	expected = "shares: invalid argument"
	c.Assert(out, checker.Contains, expected)

	out, _, err = dockerCmdWithError("run", "--cpu-shares", "99999999", "busybox", "echo", "test")
	c.Assert(err, check.NotNil, check.Commentf(out))
	expected = "The maximum allowed cpu-shares is"
	c.Assert(out, checker.Contains, expected)
}

func (s *DockerSuite) TestRunWithDefaultShmSize(c *check.C) {
	testRequires(c, DaemonIsLinux)

	name := "shm-default"
	out, _ := dockerCmd(c, "run", "--name", name, "busybox", "mount")
	shmRegex := regexp.MustCompile(`shm on /dev/shm type tmpfs(.*)size=65536k`)
	if !shmRegex.MatchString(out) {
		c.Fatalf("Expected shm of 64MB in mount command, got %v", out)
	}
	shmSize := inspectField(c, name, "HostConfig.ShmSize")
	c.Assert(shmSize, check.Equals, "67108864")
}

func (s *DockerSuite) TestRunWithShmSize(c *check.C) {
	testRequires(c, DaemonIsLinux)

	name := "shm"
	out, _ := dockerCmd(c, "run", "--name", name, "--shm-size=1G", "busybox", "mount")
	shmRegex := regexp.MustCompile(`shm on /dev/shm type tmpfs(.*)size=1048576k`)
	if !shmRegex.MatchString(out) {
		c.Fatalf("Expected shm of 1GB in mount command, got %v", out)
	}
	shmSize := inspectField(c, name, "HostConfig.ShmSize")
	c.Assert(shmSize, check.Equals, "1073741824")
}

func (s *DockerSuite) TestRunTmpfsMountsEnsureOrdered(c *check.C) {
	tmpFile, err := ioutil.TempFile("", "test")
	c.Assert(err, check.IsNil)
	defer tmpFile.Close()
	out, _ := dockerCmd(c, "run", "--tmpfs", "/run", "-v", tmpFile.Name()+":/run/test", "busybox", "ls", "/run")
	c.Assert(out, checker.Contains, "test")
}

func (s *DockerSuite) TestRunTmpfsMounts(c *check.C) {
	// TODO Windows (Post TP5): This test cannot run on a Windows daemon as
	// Windows does not support tmpfs mounts.
	testRequires(c, DaemonIsLinux)
	if out, _, err := dockerCmdWithError("run", "--tmpfs", "/run", "busybox", "touch", "/run/somefile"); err != nil {
		c.Fatalf("/run directory not mounted on tmpfs %q %s", err, out)
	}
	if out, _, err := dockerCmdWithError("run", "--tmpfs", "/run:noexec", "busybox", "touch", "/run/somefile"); err != nil {
		c.Fatalf("/run directory not mounted on tmpfs %q %s", err, out)
	}
	if out, _, err := dockerCmdWithError("run", "--tmpfs", "/run:noexec,nosuid,rw,size=5k,mode=700", "busybox", "touch", "/run/somefile"); err != nil {
		c.Fatalf("/run failed to mount on tmpfs with valid options %q %s", err, out)
	}
	if _, _, err := dockerCmdWithError("run", "--tmpfs", "/run:foobar", "busybox", "touch", "/run/somefile"); err == nil {
		c.Fatalf("/run mounted on tmpfs when it should have vailed within invalid mount option")
	}
	if _, _, err := dockerCmdWithError("run", "--tmpfs", "/run", "-v", "/run:/run", "busybox", "touch", "/run/somefile"); err == nil {
		c.Fatalf("Should have generated an error saying Duplicate mount  points")
	}
}

func (s *DockerSuite) TestRunTmpfsMountsOverrideImageVolumes(c *check.C) {
	name := "img-with-volumes"
	buildImageSuccessfully(c, name, build.WithDockerfile(`
    FROM busybox
    VOLUME /run
    RUN touch /run/stuff
    `))
	out, _ := dockerCmd(c, "run", "--tmpfs", "/run", name, "ls", "/run")
	c.Assert(out, checker.Not(checker.Contains), "stuff")
}

// Test case for #22420
func (s *DockerSuite) TestRunTmpfsMountsWithOptions(c *check.C) {
	testRequires(c, DaemonIsLinux)

	expectedOptions := []string{"rw", "nosuid", "nodev", "noexec", "relatime"}
	out, _ := dockerCmd(c, "run", "--tmpfs", "/tmp", "busybox", "sh", "-c", "mount | grep 'tmpfs on /tmp'")
	for _, option := range expectedOptions {
		c.Assert(out, checker.Contains, option)
	}
	c.Assert(out, checker.Not(checker.Contains), "size=")

	expectedOptions = []string{"rw", "nosuid", "nodev", "noexec", "relatime"}
	out, _ = dockerCmd(c, "run", "--tmpfs", "/tmp:rw", "busybox", "sh", "-c", "mount | grep 'tmpfs on /tmp'")
	for _, option := range expectedOptions {
		c.Assert(out, checker.Contains, option)
	}
	c.Assert(out, checker.Not(checker.Contains), "size=")

	expectedOptions = []string{"rw", "nosuid", "nodev", "relatime", "size=8192k"}
	out, _ = dockerCmd(c, "run", "--tmpfs", "/tmp:rw,exec,size=8192k", "busybox", "sh", "-c", "mount | grep 'tmpfs on /tmp'")
	for _, option := range expectedOptions {
		c.Assert(out, checker.Contains, option)
	}

	expectedOptions = []string{"rw", "nosuid", "nodev", "noexec", "relatime", "size=4096k"}
	out, _ = dockerCmd(c, "run", "--tmpfs", "/tmp:rw,size=8192k,exec,size=4096k,noexec", "busybox", "sh", "-c", "mount | grep 'tmpfs on /tmp'")
	for _, option := range expectedOptions {
		c.Assert(out, checker.Contains, option)
	}

	// We use debian:jessie as there is no findmnt in busybox. Also the output will be in the format of
	// TARGET PROPAGATION
	// /tmp   shared
	// so we only capture `shared` here.
	expectedOptions = []string{"shared"}
	out, _ = dockerCmd(c, "run", "--tmpfs", "/tmp:shared", "debian:jessie", "findmnt", "-o", "TARGET,PROPAGATION", "/tmp")
	for _, option := range expectedOptions {
		c.Assert(out, checker.Contains, option)
	}
}

func (s *DockerSuite) TestRunSysctls(c *check.C) {
	testRequires(c, DaemonIsLinux)
	var err error

	out, _ := dockerCmd(c, "run", "--sysctl", "net.ipv4.ip_forward=1", "--name", "test", "busybox", "cat", "/proc/sys/net/ipv4/ip_forward")
	c.Assert(strings.TrimSpace(out), check.Equals, "1")

	out = inspectFieldJSON(c, "test", "HostConfig.Sysctls")

	sysctls := make(map[string]string)
	err = json.Unmarshal([]byte(out), &sysctls)
	c.Assert(err, check.IsNil)
	c.Assert(sysctls["net.ipv4.ip_forward"], check.Equals, "1")

	out, _ = dockerCmd(c, "run", "--sysctl", "net.ipv4.ip_forward=0", "--name", "test1", "busybox", "cat", "/proc/sys/net/ipv4/ip_forward")
	c.Assert(strings.TrimSpace(out), check.Equals, "0")

	out = inspectFieldJSON(c, "test1", "HostConfig.Sysctls")

	err = json.Unmarshal([]byte(out), &sysctls)
	c.Assert(err, check.IsNil)
	c.Assert(sysctls["net.ipv4.ip_forward"], check.Equals, "0")

	icmd.RunCommand(dockerBinary, "run", "--sysctl", "kernel.foobar=1", "--name", "test2",
		"busybox", "cat", "/proc/sys/kernel/foobar").Assert(c, icmd.Expected{
		ExitCode: 125,
		Err:      "invalid argument",
	})
}

// TestRunSeccompProfileDenyUnshare checks that 'docker run --security-opt seccomp=/tmp/profile.json debian:jessie unshare' exits with operation not permitted.
func (s *DockerSuite) TestRunSeccompProfileDenyUnshare(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled, NotArm, Apparmor)
	jsonData := `{
	"defaultAction": "SCMP_ACT_ALLOW",
	"syscalls": [
		{
			"name": "unshare",
			"action": "SCMP_ACT_ERRNO"
		}
	]
}`
	tmpFile, err := ioutil.TempFile("", "profile.json")
	if err != nil {
		c.Fatal(err)
	}
	defer tmpFile.Close()

	if _, err := tmpFile.Write([]byte(jsonData)); err != nil {
		c.Fatal(err)
	}
	icmd.RunCommand(dockerBinary, "run", "--security-opt", "apparmor=unconfined",
		"--security-opt", "seccomp="+tmpFile.Name(),
		"debian:jessie", "unshare", "-p", "-m", "-f", "-r", "mount", "-t", "proc", "none", "/proc").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
}

// TestRunSeccompProfileDenyChmod checks that 'docker run --security-opt seccomp=/tmp/profile.json busybox chmod 400 /etc/hostname' exits with operation not permitted.
func (s *DockerSuite) TestRunSeccompProfileDenyChmod(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled)
	jsonData := `{
	"defaultAction": "SCMP_ACT_ALLOW",
	"syscalls": [
		{
			"name": "chmod",
			"action": "SCMP_ACT_ERRNO"
		},
		{
			"name":"fchmod",
			"action": "SCMP_ACT_ERRNO"
		},
		{
			"name": "fchmodat",
			"action":"SCMP_ACT_ERRNO"
		}
	]
}`
	tmpFile, err := ioutil.TempFile("", "profile.json")
	c.Assert(err, check.IsNil)
	defer tmpFile.Close()

	if _, err := tmpFile.Write([]byte(jsonData)); err != nil {
		c.Fatal(err)
	}
	icmd.RunCommand(dockerBinary, "run", "--security-opt", "seccomp="+tmpFile.Name(),
		"busybox", "chmod", "400", "/etc/hostname").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
}

// TestRunSeccompProfileDenyUnshareUserns checks that 'docker run debian:jessie unshare --map-root-user --user sh -c whoami' with a specific profile to
// deny unshare of a userns exits with operation not permitted.
func (s *DockerSuite) TestRunSeccompProfileDenyUnshareUserns(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled, NotArm, Apparmor)
	// from sched.h
	jsonData := fmt.Sprintf(`{
	"defaultAction": "SCMP_ACT_ALLOW",
	"syscalls": [
		{
			"name": "unshare",
			"action": "SCMP_ACT_ERRNO",
			"args": [
				{
					"index": 0,
					"value": %d,
					"op": "SCMP_CMP_EQ"
				}
			]
		}
	]
}`, uint64(0x10000000))
	tmpFile, err := ioutil.TempFile("", "profile.json")
	if err != nil {
		c.Fatal(err)
	}
	defer tmpFile.Close()

	if _, err := tmpFile.Write([]byte(jsonData)); err != nil {
		c.Fatal(err)
	}
	icmd.RunCommand(dockerBinary, "run",
		"--security-opt", "apparmor=unconfined", "--security-opt", "seccomp="+tmpFile.Name(),
		"debian:jessie", "unshare", "--map-root-user", "--user", "sh", "-c", "whoami").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
}

// TestRunSeccompProfileDenyCloneUserns checks that 'docker run syscall-test'
// with a the default seccomp profile exits with operation not permitted.
func (s *DockerSuite) TestRunSeccompProfileDenyCloneUserns(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled)
	ensureSyscallTest(c)

	icmd.RunCommand(dockerBinary, "run", "syscall-test", "userns-test", "id").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "clone failed: Operation not permitted",
	})
}

// TestRunSeccompUnconfinedCloneUserns checks that
// 'docker run --security-opt seccomp=unconfined syscall-test' allows creating a userns.
func (s *DockerSuite) TestRunSeccompUnconfinedCloneUserns(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled, UserNamespaceInKernel, NotUserNamespace, unprivilegedUsernsClone)
	ensureSyscallTest(c)

	// make sure running w privileged is ok
	icmd.RunCommand(dockerBinary, "run", "--security-opt", "seccomp=unconfined",
		"syscall-test", "userns-test", "id").Assert(c, icmd.Expected{
		Out: "nobody",
	})
}

// TestRunSeccompAllowPrivCloneUserns checks that 'docker run --privileged syscall-test'
// allows creating a userns.
func (s *DockerSuite) TestRunSeccompAllowPrivCloneUserns(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled, UserNamespaceInKernel, NotUserNamespace)
	ensureSyscallTest(c)

	// make sure running w privileged is ok
	icmd.RunCommand(dockerBinary, "run", "--privileged", "syscall-test", "userns-test", "id").Assert(c, icmd.Expected{
		Out: "nobody",
	})
}

// TestRunSeccompProfileAllow32Bit checks that 32 bit code can run on x86_64
// with the default seccomp profile.
func (s *DockerSuite) TestRunSeccompProfileAllow32Bit(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled, IsAmd64)
	ensureSyscallTest(c)

	icmd.RunCommand(dockerBinary, "run", "syscall-test", "exit32-test", "id").Assert(c, icmd.Success)
}

// TestRunSeccompAllowSetrlimit checks that 'docker run debian:jessie ulimit -v 1048510' succeeds.
func (s *DockerSuite) TestRunSeccompAllowSetrlimit(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled)

	// ulimit uses setrlimit, so we want to make sure we don't break it
	icmd.RunCommand(dockerBinary, "run", "debian:jessie", "bash", "-c", "ulimit -v 1048510").Assert(c, icmd.Success)
}

func (s *DockerSuite) TestRunSeccompDefaultProfileAcct(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled, NotUserNamespace)
	ensureSyscallTest(c)

	out, _, err := dockerCmdWithError("run", "syscall-test", "acct-test")
	if err == nil || !strings.Contains(out, "Operation not permitted") {
		c.Fatalf("test 0: expected Operation not permitted, got: %s", out)
	}

	out, _, err = dockerCmdWithError("run", "--cap-add", "sys_admin", "syscall-test", "acct-test")
	if err == nil || !strings.Contains(out, "Operation not permitted") {
		c.Fatalf("test 1: expected Operation not permitted, got: %s", out)
	}

	out, _, err = dockerCmdWithError("run", "--cap-add", "sys_pacct", "syscall-test", "acct-test")
	if err == nil || !strings.Contains(out, "No such file or directory") {
		c.Fatalf("test 2: expected No such file or directory, got: %s", out)
	}

	out, _, err = dockerCmdWithError("run", "--cap-add", "ALL", "syscall-test", "acct-test")
	if err == nil || !strings.Contains(out, "No such file or directory") {
		c.Fatalf("test 3: expected No such file or directory, got: %s", out)
	}

	out, _, err = dockerCmdWithError("run", "--cap-drop", "ALL", "--cap-add", "sys_pacct", "syscall-test", "acct-test")
	if err == nil || !strings.Contains(out, "No such file or directory") {
		c.Fatalf("test 4: expected No such file or directory, got: %s", out)
	}
}

func (s *DockerSuite) TestRunSeccompDefaultProfileNS(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled, NotUserNamespace)
	ensureSyscallTest(c)

	out, _, err := dockerCmdWithError("run", "syscall-test", "ns-test", "echo", "hello0")
	if err == nil || !strings.Contains(out, "Operation not permitted") {
		c.Fatalf("test 0: expected Operation not permitted, got: %s", out)
	}

	out, _, err = dockerCmdWithError("run", "--cap-add", "sys_admin", "syscall-test", "ns-test", "echo", "hello1")
	if err != nil || !strings.Contains(out, "hello1") {
		c.Fatalf("test 1: expected hello1, got: %s, %v", out, err)
	}

	out, _, err = dockerCmdWithError("run", "--cap-drop", "all", "--cap-add", "sys_admin", "syscall-test", "ns-test", "echo", "hello2")
	if err != nil || !strings.Contains(out, "hello2") {
		c.Fatalf("test 2: expected hello2, got: %s, %v", out, err)
	}

	out, _, err = dockerCmdWithError("run", "--cap-add", "ALL", "syscall-test", "ns-test", "echo", "hello3")
	if err != nil || !strings.Contains(out, "hello3") {
		c.Fatalf("test 3: expected hello3, got: %s, %v", out, err)
	}

	out, _, err = dockerCmdWithError("run", "--cap-add", "ALL", "--security-opt", "seccomp=unconfined", "syscall-test", "acct-test")
	if err == nil || !strings.Contains(out, "No such file or directory") {
		c.Fatalf("test 4: expected No such file or directory, got: %s", out)
	}

	out, _, err = dockerCmdWithError("run", "--cap-add", "ALL", "--security-opt", "seccomp=unconfined", "syscall-test", "ns-test", "echo", "hello4")
	if err != nil || !strings.Contains(out, "hello4") {
		c.Fatalf("test 5: expected hello4, got: %s, %v", out, err)
	}
}

// TestRunNoNewPrivSetuid checks that --security-opt='no-new-privileges=true' prevents
// effective uid transtions on executing setuid binaries.
func (s *DockerSuite) TestRunNoNewPrivSetuid(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace, SameHostDaemon)
	ensureNNPTest(c)

	// test that running a setuid binary results in no effective uid transition
	icmd.RunCommand(dockerBinary, "run", "--security-opt", "no-new-privileges=true", "--user", "1000",
		"nnp-test", "/usr/bin/nnp-test").Assert(c, icmd.Expected{
		Out: "EUID=1000",
	})
}

// TestLegacyRunNoNewPrivSetuid checks that --security-opt=no-new-privileges prevents
// effective uid transtions on executing setuid binaries.
func (s *DockerSuite) TestLegacyRunNoNewPrivSetuid(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace, SameHostDaemon)
	ensureNNPTest(c)

	// test that running a setuid binary results in no effective uid transition
	icmd.RunCommand(dockerBinary, "run", "--security-opt", "no-new-privileges", "--user", "1000",
		"nnp-test", "/usr/bin/nnp-test").Assert(c, icmd.Expected{
		Out: "EUID=1000",
	})
}

func (s *DockerSuite) TestUserNoEffectiveCapabilitiesChown(c *check.C) {
	testRequires(c, DaemonIsLinux)
	ensureSyscallTest(c)

	// test that a root user has default capability CAP_CHOWN
	dockerCmd(c, "run", "busybox", "chown", "100", "/tmp")
	// test that non root user does not have default capability CAP_CHOWN
	icmd.RunCommand(dockerBinary, "run", "--user", "1000:1000", "busybox", "chown", "100", "/tmp").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
	// test that root user can drop default capability CAP_CHOWN
	icmd.RunCommand(dockerBinary, "run", "--cap-drop", "chown", "busybox", "chown", "100", "/tmp").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
}

func (s *DockerSuite) TestUserNoEffectiveCapabilitiesDacOverride(c *check.C) {
	testRequires(c, DaemonIsLinux)
	ensureSyscallTest(c)

	// test that a root user has default capability CAP_DAC_OVERRIDE
	dockerCmd(c, "run", "busybox", "sh", "-c", "echo test > /etc/passwd")
	// test that non root user does not have default capability CAP_DAC_OVERRIDE
	icmd.RunCommand(dockerBinary, "run", "--user", "1000:1000", "busybox", "sh", "-c", "echo test > /etc/passwd").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Permission denied",
	})
}

func (s *DockerSuite) TestUserNoEffectiveCapabilitiesFowner(c *check.C) {
	testRequires(c, DaemonIsLinux)
	ensureSyscallTest(c)

	// test that a root user has default capability CAP_FOWNER
	dockerCmd(c, "run", "busybox", "chmod", "777", "/etc/passwd")
	// test that non root user does not have default capability CAP_FOWNER
	icmd.RunCommand(dockerBinary, "run", "--user", "1000:1000", "busybox", "chmod", "777", "/etc/passwd").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
	// TODO test that root user can drop default capability CAP_FOWNER
}

// TODO CAP_KILL

func (s *DockerSuite) TestUserNoEffectiveCapabilitiesSetuid(c *check.C) {
	testRequires(c, DaemonIsLinux)
	ensureSyscallTest(c)

	// test that a root user has default capability CAP_SETUID
	dockerCmd(c, "run", "syscall-test", "setuid-test")
	// test that non root user does not have default capability CAP_SETUID
	icmd.RunCommand(dockerBinary, "run", "--user", "1000:1000", "syscall-test", "setuid-test").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
	// test that root user can drop default capability CAP_SETUID
	icmd.RunCommand(dockerBinary, "run", "--cap-drop", "setuid", "syscall-test", "setuid-test").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
}

func (s *DockerSuite) TestUserNoEffectiveCapabilitiesSetgid(c *check.C) {
	testRequires(c, DaemonIsLinux)
	ensureSyscallTest(c)

	// test that a root user has default capability CAP_SETGID
	dockerCmd(c, "run", "syscall-test", "setgid-test")
	// test that non root user does not have default capability CAP_SETGID
	icmd.RunCommand(dockerBinary, "run", "--user", "1000:1000", "syscall-test", "setgid-test").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
	// test that root user can drop default capability CAP_SETGID
	icmd.RunCommand(dockerBinary, "run", "--cap-drop", "setgid", "syscall-test", "setgid-test").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
}

// TODO CAP_SETPCAP

func (s *DockerSuite) TestUserNoEffectiveCapabilitiesNetBindService(c *check.C) {
	testRequires(c, DaemonIsLinux)
	ensureSyscallTest(c)

	// test that a root user has default capability CAP_NET_BIND_SERVICE
	dockerCmd(c, "run", "syscall-test", "socket-test")
	// test that non root user does not have default capability CAP_NET_BIND_SERVICE
	icmd.RunCommand(dockerBinary, "run", "--user", "1000:1000", "syscall-test", "socket-test").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Permission denied",
	})
	// test that root user can drop default capability CAP_NET_BIND_SERVICE
	icmd.RunCommand(dockerBinary, "run", "--cap-drop", "net_bind_service", "syscall-test", "socket-test").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Permission denied",
	})
}

func (s *DockerSuite) TestUserNoEffectiveCapabilitiesNetRaw(c *check.C) {
	testRequires(c, DaemonIsLinux)
	ensureSyscallTest(c)

	// test that a root user has default capability CAP_NET_RAW
	dockerCmd(c, "run", "syscall-test", "raw-test")
	// test that non root user does not have default capability CAP_NET_RAW
	icmd.RunCommand(dockerBinary, "run", "--user", "1000:1000", "syscall-test", "raw-test").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
	// test that root user can drop default capability CAP_NET_RAW
	icmd.RunCommand(dockerBinary, "run", "--cap-drop", "net_raw", "syscall-test", "raw-test").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
}

func (s *DockerSuite) TestUserNoEffectiveCapabilitiesChroot(c *check.C) {
	testRequires(c, DaemonIsLinux)
	ensureSyscallTest(c)

	// test that a root user has default capability CAP_SYS_CHROOT
	dockerCmd(c, "run", "busybox", "chroot", "/", "/bin/true")
	// test that non root user does not have default capability CAP_SYS_CHROOT
	icmd.RunCommand(dockerBinary, "run", "--user", "1000:1000", "busybox", "chroot", "/", "/bin/true").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
	// test that root user can drop default capability CAP_SYS_CHROOT
	icmd.RunCommand(dockerBinary, "run", "--cap-drop", "sys_chroot", "busybox", "chroot", "/", "/bin/true").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
}

func (s *DockerSuite) TestUserNoEffectiveCapabilitiesMknod(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	ensureSyscallTest(c)

	// test that a root user has default capability CAP_MKNOD
	dockerCmd(c, "run", "busybox", "mknod", "/tmp/node", "b", "1", "2")
	// test that non root user does not have default capability CAP_MKNOD
	// test that root user can drop default capability CAP_SYS_CHROOT
	icmd.RunCommand(dockerBinary, "run", "--user", "1000:1000", "busybox", "mknod", "/tmp/node", "b", "1", "2").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
	// test that root user can drop default capability CAP_MKNOD
	icmd.RunCommand(dockerBinary, "run", "--cap-drop", "mknod", "busybox", "mknod", "/tmp/node", "b", "1", "2").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
}

// TODO CAP_AUDIT_WRITE
// TODO CAP_SETFCAP

func (s *DockerSuite) TestRunApparmorProcDirectory(c *check.C) {
	testRequires(c, SameHostDaemon, Apparmor)

	// running w seccomp unconfined tests the apparmor profile
	result := icmd.RunCommand(dockerBinary, "run", "--security-opt", "seccomp=unconfined", "busybox", "chmod", "777", "/proc/1/cgroup")
	result.Assert(c, icmd.Expected{ExitCode: 1})
	if !(strings.Contains(result.Combined(), "Permission denied") || strings.Contains(result.Combined(), "Operation not permitted")) {
		c.Fatalf("expected chmod 777 /proc/1/cgroup to fail, got %s: %v", result.Combined(), result.Error)
	}

	result = icmd.RunCommand(dockerBinary, "run", "--security-opt", "seccomp=unconfined", "busybox", "chmod", "777", "/proc/1/attr/current")
	result.Assert(c, icmd.Expected{ExitCode: 1})
	if !(strings.Contains(result.Combined(), "Permission denied") || strings.Contains(result.Combined(), "Operation not permitted")) {
		c.Fatalf("expected chmod 777 /proc/1/attr/current to fail, got %s: %v", result.Combined(), result.Error)
	}
}

// make sure the default profile can be successfully parsed (using unshare as it is
// something which we know is blocked in the default profile)
func (s *DockerSuite) TestRunSeccompWithDefaultProfile(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled)

	out, _, err := dockerCmdWithError("run", "--security-opt", "seccomp=../profiles/seccomp/default.json", "debian:jessie", "unshare", "--map-root-user", "--user", "sh", "-c", "whoami")
	c.Assert(err, checker.NotNil, check.Commentf(out))
	c.Assert(strings.TrimSpace(out), checker.Equals, "unshare: unshare failed: Operation not permitted")
}

// TestRunDeviceSymlink checks run with device that follows symlink (#13840 and #22271)
func (s *DockerSuite) TestRunDeviceSymlink(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace, NotArm, SameHostDaemon)
	if _, err := os.Stat("/dev/zero"); err != nil {
		c.Skip("Host does not have /dev/zero")
	}

	// Create a temporary directory to create symlink
	tmpDir, err := ioutil.TempDir("", "docker_device_follow_symlink_tests")
	c.Assert(err, checker.IsNil)

	defer os.RemoveAll(tmpDir)

	// Create a symbolic link to /dev/zero
	symZero := filepath.Join(tmpDir, "zero")
	err = os.Symlink("/dev/zero", symZero)
	c.Assert(err, checker.IsNil)

	// Create a temporary file "temp" inside tmpDir, write some data to "tmpDir/temp",
	// then create a symlink "tmpDir/file" to the temporary file "tmpDir/temp".
	tmpFile := filepath.Join(tmpDir, "temp")
	err = ioutil.WriteFile(tmpFile, []byte("temp"), 0666)
	c.Assert(err, checker.IsNil)
	symFile := filepath.Join(tmpDir, "file")
	err = os.Symlink(tmpFile, symFile)
	c.Assert(err, checker.IsNil)

	// Create a symbolic link to /dev/zero, this time with a relative path (#22271)
	err = os.Symlink("zero", "/dev/symzero")
	if err != nil {
		c.Fatal("/dev/symzero creation failed")
	}
	// We need to remove this symbolic link here as it is created in /dev/, not temporary directory as above
	defer os.Remove("/dev/symzero")

	// md5sum of 'dd if=/dev/zero bs=4K count=8' is bb7df04e1b0a2570657527a7e108ae23
	out, _ := dockerCmd(c, "run", "--device", symZero+":/dev/symzero", "busybox", "sh", "-c", "dd if=/dev/symzero bs=4K count=8 | md5sum")
	c.Assert(strings.Trim(out, "\r\n"), checker.Contains, "bb7df04e1b0a2570657527a7e108ae23", check.Commentf("expected output bb7df04e1b0a2570657527a7e108ae23"))

	// symlink "tmpDir/file" to a file "tmpDir/temp" will result in an error as it is not a device.
	out, _, err = dockerCmdWithError("run", "--device", symFile+":/dev/symzero", "busybox", "sh", "-c", "dd if=/dev/symzero bs=4K count=8 | md5sum")
	c.Assert(err, check.NotNil)
	c.Assert(strings.Trim(out, "\r\n"), checker.Contains, "not a device node", check.Commentf("expected output 'not a device node'"))

	// md5sum of 'dd if=/dev/zero bs=4K count=8' is bb7df04e1b0a2570657527a7e108ae23 (this time check with relative path backed, see #22271)
	out, _ = dockerCmd(c, "run", "--device", "/dev/symzero:/dev/symzero", "busybox", "sh", "-c", "dd if=/dev/symzero bs=4K count=8 | md5sum")
	c.Assert(strings.Trim(out, "\r\n"), checker.Contains, "bb7df04e1b0a2570657527a7e108ae23", check.Commentf("expected output bb7df04e1b0a2570657527a7e108ae23"))
}

// TestRunPIDsLimit makes sure the pids cgroup is set with --pids-limit
func (s *DockerSuite) TestRunPIDsLimit(c *check.C) {
	testRequires(c, pidsLimit)

	file := "/sys/fs/cgroup/pids/pids.max"
	out, _ := dockerCmd(c, "run", "--name", "skittles", "--pids-limit", "4", "busybox", "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "4")

	out = inspectField(c, "skittles", "HostConfig.PidsLimit")
	c.Assert(out, checker.Equals, "4", check.Commentf("setting the pids limit failed"))
}

func (s *DockerSuite) TestRunPrivilegedAllowedDevices(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)

	file := "/sys/fs/cgroup/devices/devices.list"
	out, _ := dockerCmd(c, "run", "--privileged", "busybox", "cat", file)
	c.Logf("out: %q", out)
	c.Assert(strings.TrimSpace(out), checker.Equals, "a *:* rwm")
}

func (s *DockerSuite) TestRunUserDeviceAllowed(c *check.C) {
	testRequires(c, DaemonIsLinux)

	fi, err := os.Stat("/dev/snd/timer")
	if err != nil {
		c.Skip("Host does not have /dev/snd/timer")
	}
	stat, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		c.Skip("Could not stat /dev/snd/timer")
	}

	file := "/sys/fs/cgroup/devices/devices.list"
	out, _ := dockerCmd(c, "run", "--device", "/dev/snd/timer:w", "busybox", "cat", file)
	c.Assert(out, checker.Contains, fmt.Sprintf("c %d:%d w", stat.Rdev/256, stat.Rdev%256))
}

func (s *DockerDaemonSuite) TestRunSeccompJSONNewFormat(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled)

	s.d.StartWithBusybox(c)

	jsonData := `{
	"defaultAction": "SCMP_ACT_ALLOW",
	"syscalls": [
		{
			"names": ["chmod", "fchmod", "fchmodat"],
			"action": "SCMP_ACT_ERRNO"
		}
	]
}`
	tmpFile, err := ioutil.TempFile("", "profile.json")
	c.Assert(err, check.IsNil)
	defer tmpFile.Close()
	_, err = tmpFile.Write([]byte(jsonData))
	c.Assert(err, check.IsNil)

	out, err := s.d.Cmd("run", "--security-opt", "seccomp="+tmpFile.Name(), "busybox", "chmod", "777", ".")
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, "Operation not permitted")
}

func (s *DockerDaemonSuite) TestRunSeccompJSONNoNameAndNames(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled)

	s.d.StartWithBusybox(c)

	jsonData := `{
	"defaultAction": "SCMP_ACT_ALLOW",
	"syscalls": [
		{
			"name": "chmod",
			"names": ["fchmod", "fchmodat"],
			"action": "SCMP_ACT_ERRNO"
		}
	]
}`
	tmpFile, err := ioutil.TempFile("", "profile.json")
	c.Assert(err, check.IsNil)
	defer tmpFile.Close()
	_, err = tmpFile.Write([]byte(jsonData))
	c.Assert(err, check.IsNil)

	out, err := s.d.Cmd("run", "--security-opt", "seccomp="+tmpFile.Name(), "busybox", "chmod", "777", ".")
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, "'name' and 'names' were specified in the seccomp profile, use either 'name' or 'names'")
}

func (s *DockerDaemonSuite) TestRunSeccompJSONNoArchAndArchMap(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled)

	s.d.StartWithBusybox(c)

	jsonData := `{
	"archMap": [
		{
			"architecture": "SCMP_ARCH_X86_64",
			"subArchitectures": [
				"SCMP_ARCH_X86",
				"SCMP_ARCH_X32"
			]
		}
	],
	"architectures": [
		"SCMP_ARCH_X32"
	],
	"defaultAction": "SCMP_ACT_ALLOW",
	"syscalls": [
		{
			"names": ["chmod", "fchmod", "fchmodat"],
			"action": "SCMP_ACT_ERRNO"
		}
	]
}`
	tmpFile, err := ioutil.TempFile("", "profile.json")
	c.Assert(err, check.IsNil)
	defer tmpFile.Close()
	_, err = tmpFile.Write([]byte(jsonData))
	c.Assert(err, check.IsNil)

	out, err := s.d.Cmd("run", "--security-opt", "seccomp="+tmpFile.Name(), "busybox", "chmod", "777", ".")
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, "'architectures' and 'archMap' were specified in the seccomp profile, use either 'architectures' or 'archMap'")
}

func (s *DockerDaemonSuite) TestRunWithDaemonDefaultSeccompProfile(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled)

	s.d.StartWithBusybox(c)

	// 1) verify I can run containers with the Docker default shipped profile which allows chmod
	_, err := s.d.Cmd("run", "busybox", "chmod", "777", ".")
	c.Assert(err, check.IsNil)

	jsonData := `{
	"defaultAction": "SCMP_ACT_ALLOW",
	"syscalls": [
		{
			"name": "chmod",
			"action": "SCMP_ACT_ERRNO"
		}
	]
}`
	tmpFile, err := ioutil.TempFile("", "profile.json")
	c.Assert(err, check.IsNil)
	defer tmpFile.Close()
	_, err = tmpFile.Write([]byte(jsonData))
	c.Assert(err, check.IsNil)

	// 2) restart the daemon and add a custom seccomp profile in which we deny chmod
	s.d.Restart(c, "--seccomp-profile="+tmpFile.Name())

	out, err := s.d.Cmd("run", "busybox", "chmod", "777", ".")
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, "Operation not permitted")
}

func (s *DockerSuite) TestRunWithNanoCPUs(c *check.C) {
	testRequires(c, cpuCfsQuota, cpuCfsPeriod)

	file1 := "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
	file2 := "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
	out, _ := dockerCmd(c, "run", "--cpus", "0.5", "--name", "test", "busybox", "sh", "-c", fmt.Sprintf("cat %s && cat %s", file1, file2))
	c.Assert(strings.TrimSpace(out), checker.Equals, "50000\n100000")

	out = inspectField(c, "test", "HostConfig.NanoCpus")
	c.Assert(out, checker.Equals, "5e+08", check.Commentf("setting the Nano CPUs failed"))
	out = inspectField(c, "test", "HostConfig.CpuQuota")
	c.Assert(out, checker.Equals, "0", check.Commentf("CPU CFS quota should be 0"))
	out = inspectField(c, "test", "HostConfig.CpuPeriod")
	c.Assert(out, checker.Equals, "0", check.Commentf("CPU CFS period should be 0"))

	out, _, err := dockerCmdWithError("run", "--cpus", "0.5", "--cpu-quota", "50000", "--cpu-period", "100000", "busybox", "sh")
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, "Conflicting options: Nano CPUs and CPU Period cannot both be set")
}
