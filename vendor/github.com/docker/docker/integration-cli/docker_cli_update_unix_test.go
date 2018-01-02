// +build !windows

package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/pkg/parsers/kernel"
	"github.com/go-check/check"
	"github.com/kr/pty"
)

func (s *DockerSuite) TestUpdateRunningContainer(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "-m", "300M", "busybox", "top")
	dockerCmd(c, "update", "-m", "500M", name)

	c.Assert(inspectField(c, name, "HostConfig.Memory"), checker.Equals, "524288000")

	file := "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	out, _ := dockerCmd(c, "exec", name, "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "524288000")
}

func (s *DockerSuite) TestUpdateRunningContainerWithRestart(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "-m", "300M", "busybox", "top")
	dockerCmd(c, "update", "-m", "500M", name)
	dockerCmd(c, "restart", name)

	c.Assert(inspectField(c, name, "HostConfig.Memory"), checker.Equals, "524288000")

	file := "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	out, _ := dockerCmd(c, "exec", name, "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "524288000")
}

func (s *DockerSuite) TestUpdateStoppedContainer(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)

	name := "test-update-container"
	file := "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	dockerCmd(c, "run", "--name", name, "-m", "300M", "busybox", "cat", file)
	dockerCmd(c, "update", "-m", "500M", name)

	c.Assert(inspectField(c, name, "HostConfig.Memory"), checker.Equals, "524288000")

	out, _ := dockerCmd(c, "start", "-a", name)
	c.Assert(strings.TrimSpace(out), checker.Equals, "524288000")
}

func (s *DockerSuite) TestUpdatePausedContainer(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, cpuShare)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "--cpu-shares", "1000", "busybox", "top")
	dockerCmd(c, "pause", name)
	dockerCmd(c, "update", "--cpu-shares", "500", name)

	c.Assert(inspectField(c, name, "HostConfig.CPUShares"), checker.Equals, "500")

	dockerCmd(c, "unpause", name)
	file := "/sys/fs/cgroup/cpu/cpu.shares"
	out, _ := dockerCmd(c, "exec", name, "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "500")
}

func (s *DockerSuite) TestUpdateWithUntouchedFields(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)
	testRequires(c, cpuShare)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "-m", "300M", "--cpu-shares", "800", "busybox", "top")
	dockerCmd(c, "update", "-m", "500M", name)

	// Update memory and not touch cpus, `cpuset.cpus` should still have the old value
	out := inspectField(c, name, "HostConfig.CPUShares")
	c.Assert(out, check.Equals, "800")

	file := "/sys/fs/cgroup/cpu/cpu.shares"
	out, _ = dockerCmd(c, "exec", name, "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "800")
}

func (s *DockerSuite) TestUpdateContainerInvalidValue(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "-m", "300M", "busybox", "true")
	out, _, err := dockerCmdWithError("update", "-m", "2M", name)
	c.Assert(err, check.NotNil)
	expected := "Minimum memory limit allowed is 4MB"
	c.Assert(out, checker.Contains, expected)
}

func (s *DockerSuite) TestUpdateContainerWithoutFlags(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "-m", "300M", "busybox", "true")
	_, _, err := dockerCmdWithError("update", name)
	c.Assert(err, check.NotNil)
}

func (s *DockerSuite) TestUpdateKernelMemory(c *check.C) {
	testRequires(c, DaemonIsLinux, kernelMemorySupport)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "--kernel-memory", "50M", "busybox", "top")
	dockerCmd(c, "update", "--kernel-memory", "100M", name)

	c.Assert(inspectField(c, name, "HostConfig.KernelMemory"), checker.Equals, "104857600")

	file := "/sys/fs/cgroup/memory/memory.kmem.limit_in_bytes"
	out, _ := dockerCmd(c, "exec", name, "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "104857600")
}

func (s *DockerSuite) TestUpdateKernelMemoryUninitialized(c *check.C) {
	testRequires(c, DaemonIsLinux, kernelMemorySupport)

	isNewKernel := kernel.CheckKernelVersion(4, 6, 0)
	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "busybox", "top")
	_, _, err := dockerCmdWithError("update", "--kernel-memory", "100M", name)
	// Update kernel memory to a running container without kernel memory initialized
	// is not allowed before kernel version 4.6.
	if !isNewKernel {
		c.Assert(err, check.NotNil)
	} else {
		c.Assert(err, check.IsNil)
	}

	dockerCmd(c, "pause", name)
	_, _, err = dockerCmdWithError("update", "--kernel-memory", "200M", name)
	if !isNewKernel {
		c.Assert(err, check.NotNil)
	} else {
		c.Assert(err, check.IsNil)
	}
	dockerCmd(c, "unpause", name)

	dockerCmd(c, "stop", name)
	dockerCmd(c, "update", "--kernel-memory", "300M", name)
	dockerCmd(c, "start", name)

	c.Assert(inspectField(c, name, "HostConfig.KernelMemory"), checker.Equals, "314572800")

	file := "/sys/fs/cgroup/memory/memory.kmem.limit_in_bytes"
	out, _ := dockerCmd(c, "exec", name, "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "314572800")
}

func (s *DockerSuite) TestUpdateSwapMemoryOnly(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)
	testRequires(c, swapMemorySupport)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "--memory", "300M", "--memory-swap", "500M", "busybox", "top")
	dockerCmd(c, "update", "--memory-swap", "600M", name)

	c.Assert(inspectField(c, name, "HostConfig.MemorySwap"), checker.Equals, "629145600")

	file := "/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes"
	out, _ := dockerCmd(c, "exec", name, "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "629145600")
}

func (s *DockerSuite) TestUpdateInvalidSwapMemory(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)
	testRequires(c, swapMemorySupport)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "--memory", "300M", "--memory-swap", "500M", "busybox", "top")
	_, _, err := dockerCmdWithError("update", "--memory-swap", "200M", name)
	// Update invalid swap memory should fail.
	// This will pass docker config validation, but failed at kernel validation
	c.Assert(err, check.NotNil)

	// Update invalid swap memory with failure should not change HostConfig
	c.Assert(inspectField(c, name, "HostConfig.Memory"), checker.Equals, "314572800")
	c.Assert(inspectField(c, name, "HostConfig.MemorySwap"), checker.Equals, "524288000")

	dockerCmd(c, "update", "--memory-swap", "600M", name)

	c.Assert(inspectField(c, name, "HostConfig.MemorySwap"), checker.Equals, "629145600")

	file := "/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes"
	out, _ := dockerCmd(c, "exec", name, "cat", file)
	c.Assert(strings.TrimSpace(out), checker.Equals, "629145600")
}

func (s *DockerSuite) TestUpdateStats(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)
	testRequires(c, cpuCfsQuota)
	name := "foo"
	dockerCmd(c, "run", "-d", "-ti", "--name", name, "-m", "500m", "busybox")

	c.Assert(waitRun(name), checker.IsNil)

	getMemLimit := func(id string) uint64 {
		resp, body, err := request.Get(fmt.Sprintf("/containers/%s/stats?stream=false", id))
		c.Assert(err, checker.IsNil)
		c.Assert(resp.Header.Get("Content-Type"), checker.Equals, "application/json")

		var v *types.Stats
		err = json.NewDecoder(body).Decode(&v)
		c.Assert(err, checker.IsNil)
		body.Close()

		return v.MemoryStats.Limit
	}
	preMemLimit := getMemLimit(name)

	dockerCmd(c, "update", "--cpu-quota", "2000", name)

	curMemLimit := getMemLimit(name)

	c.Assert(preMemLimit, checker.Equals, curMemLimit)

}

func (s *DockerSuite) TestUpdateMemoryWithSwapMemory(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, memoryLimitSupport)
	testRequires(c, swapMemorySupport)

	name := "test-update-container"
	dockerCmd(c, "run", "-d", "--name", name, "--memory", "300M", "busybox", "top")
	out, _, err := dockerCmdWithError("update", "--memory", "800M", name)
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "Memory limit should be smaller than already set memoryswap limit")

	dockerCmd(c, "update", "--memory", "800M", "--memory-swap", "1000M", name)
}

func (s *DockerSuite) TestUpdateNotAffectMonitorRestartPolicy(c *check.C) {
	testRequires(c, DaemonIsLinux, cpuShare)

	out, _ := dockerCmd(c, "run", "-tid", "--restart=always", "busybox", "sh")
	id := strings.TrimSpace(string(out))
	dockerCmd(c, "update", "--cpu-shares", "512", id)

	cpty, tty, err := pty.Open()
	c.Assert(err, checker.IsNil)
	defer cpty.Close()

	cmd := exec.Command(dockerBinary, "attach", id)
	cmd.Stdin = tty

	c.Assert(cmd.Start(), checker.IsNil)
	defer cmd.Process.Kill()

	_, err = cpty.Write([]byte("exit\n"))
	c.Assert(err, checker.IsNil)

	c.Assert(cmd.Wait(), checker.IsNil)

	// container should restart again and keep running
	err = waitInspect(id, "{{.RestartCount}}", "1", 30*time.Second)
	c.Assert(err, checker.IsNil)
	c.Assert(waitRun(id), checker.IsNil)
}

func (s *DockerSuite) TestUpdateWithNanoCPUs(c *check.C) {
	testRequires(c, cpuCfsQuota, cpuCfsPeriod)

	file1 := "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
	file2 := "/sys/fs/cgroup/cpu/cpu.cfs_period_us"

	out, _ := dockerCmd(c, "run", "-d", "--cpus", "0.5", "--name", "top", "busybox", "top")
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), "")

	out, _ = dockerCmd(c, "exec", "top", "sh", "-c", fmt.Sprintf("cat %s && cat %s", file1, file2))
	c.Assert(strings.TrimSpace(out), checker.Equals, "50000\n100000")

	out = inspectField(c, "top", "HostConfig.NanoCpus")
	c.Assert(out, checker.Equals, "5e+08", check.Commentf("setting the Nano CPUs failed"))
	out = inspectField(c, "top", "HostConfig.CpuQuota")
	c.Assert(out, checker.Equals, "0", check.Commentf("CPU CFS quota should be 0"))
	out = inspectField(c, "top", "HostConfig.CpuPeriod")
	c.Assert(out, checker.Equals, "0", check.Commentf("CPU CFS period should be 0"))

	out, _, err := dockerCmdWithError("update", "--cpu-quota", "80000", "top")
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "Conflicting options: CPU Quota cannot be updated as NanoCPUs has already been set")

	out, _ = dockerCmd(c, "update", "--cpus", "0.8", "top")
	out = inspectField(c, "top", "HostConfig.NanoCpus")
	c.Assert(out, checker.Equals, "8e+08", check.Commentf("updating the Nano CPUs failed"))
	out = inspectField(c, "top", "HostConfig.CpuQuota")
	c.Assert(out, checker.Equals, "0", check.Commentf("CPU CFS quota should be 0"))
	out = inspectField(c, "top", "HostConfig.CpuPeriod")
	c.Assert(out, checker.Equals, "0", check.Commentf("CPU CFS period should be 0"))

	out, _ = dockerCmd(c, "exec", "top", "sh", "-c", fmt.Sprintf("cat %s && cat %s", file1, file2))
	c.Assert(strings.TrimSpace(out), checker.Equals, "80000\n100000")
}
