package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/cgroups/systemd"
	"github.com/opencontainers/runc/libcontainer/configs"

	"golang.org/x/sys/unix"
)

func TestExecPS(t *testing.T) {
	testExecPS(t, false)
}

func TestUsernsExecPS(t *testing.T) {
	if _, err := os.Stat("/proc/self/ns/user"); os.IsNotExist(err) {
		t.Skip("userns is unsupported")
	}
	testExecPS(t, true)
}

func testExecPS(t *testing.T, userns bool) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)
	if userns {
		config.UidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
		config.GidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
		config.Namespaces = append(config.Namespaces, configs.Namespace{Type: configs.NEWUSER})
	}

	buffers, exitCode, err := runContainer(config, "", "ps", "-o", "pid,user,comm")
	if err != nil {
		t.Fatalf("%s: %s", buffers, err)
	}
	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}
	lines := strings.Split(buffers.Stdout.String(), "\n")
	if len(lines) < 2 {
		t.Fatalf("more than one process running for output %q", buffers.Stdout.String())
	}
	expected := `1 root     ps`
	actual := strings.Trim(lines[1], "\n ")
	if actual != expected {
		t.Fatalf("expected output %q but received %q", expected, actual)
	}
}

func TestIPCPrivate(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	l, err := os.Readlink("/proc/1/ns/ipc")
	ok(t, err)

	config := newTemplateConfig(rootfs)
	buffers, exitCode, err := runContainer(config, "", "readlink", "/proc/self/ns/ipc")
	ok(t, err)

	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}

	if actual := strings.Trim(buffers.Stdout.String(), "\n"); actual == l {
		t.Fatalf("ipc link should be private to the container but equals host %q %q", actual, l)
	}
}

func TestIPCHost(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	l, err := os.Readlink("/proc/1/ns/ipc")
	ok(t, err)

	config := newTemplateConfig(rootfs)
	config.Namespaces.Remove(configs.NEWIPC)
	buffers, exitCode, err := runContainer(config, "", "readlink", "/proc/self/ns/ipc")
	ok(t, err)

	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}

	if actual := strings.Trim(buffers.Stdout.String(), "\n"); actual != l {
		t.Fatalf("ipc link not equal to host link %q %q", actual, l)
	}
}

func TestIPCJoinPath(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	l, err := os.Readlink("/proc/1/ns/ipc")
	ok(t, err)

	config := newTemplateConfig(rootfs)
	config.Namespaces.Add(configs.NEWIPC, "/proc/1/ns/ipc")

	buffers, exitCode, err := runContainer(config, "", "readlink", "/proc/self/ns/ipc")
	ok(t, err)

	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}

	if actual := strings.Trim(buffers.Stdout.String(), "\n"); actual != l {
		t.Fatalf("ipc link not equal to host link %q %q", actual, l)
	}
}

func TestIPCBadPath(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	config.Namespaces.Add(configs.NEWIPC, "/proc/1/ns/ipcc")

	_, _, err = runContainer(config, "", "true")
	if err == nil {
		t.Fatal("container succeeded with bad ipc path")
	}
}

func TestRlimit(t *testing.T) {
	testRlimit(t, false)
}

func TestUsernsRlimit(t *testing.T) {
	if _, err := os.Stat("/proc/self/ns/user"); os.IsNotExist(err) {
		t.Skip("userns is unsupported")
	}

	testRlimit(t, true)
}

func testRlimit(t *testing.T, userns bool) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	if userns {
		config.UidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
		config.GidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
		config.Namespaces = append(config.Namespaces, configs.Namespace{Type: configs.NEWUSER})
	}

	// ensure limit is lower than what the config requests to test that in a user namespace
	// the Setrlimit call happens early enough that we still have permissions to raise the limit.
	ok(t, unix.Setrlimit(unix.RLIMIT_NOFILE, &unix.Rlimit{
		Max: 1024,
		Cur: 1024,
	}))

	out, _, err := runContainer(config, "", "/bin/sh", "-c", "ulimit -n")
	ok(t, err)
	if limit := strings.TrimSpace(out.Stdout.String()); limit != "1025" {
		t.Fatalf("expected rlimit to be 1025, got %s", limit)
	}
}

func TestEnter(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	ok(t, err)
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)

	container, err := factory.Create("test", config)
	ok(t, err)
	defer container.Destroy()

	// Execute a first process in the container
	stdinR, stdinW, err := os.Pipe()
	ok(t, err)

	var stdout, stdout2 bytes.Buffer

	pconfig := libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"sh", "-c", "cat && readlink /proc/self/ns/pid"},
		Env:    standardEnvironment,
		Stdin:  stdinR,
		Stdout: &stdout,
	}
	err = container.Run(&pconfig)
	stdinR.Close()
	defer stdinW.Close()
	ok(t, err)
	pid, err := pconfig.Pid()
	ok(t, err)

	// Execute another process in the container
	stdinR2, stdinW2, err := os.Pipe()
	ok(t, err)
	pconfig2 := libcontainer.Process{
		Cwd: "/",
		Env: standardEnvironment,
	}
	pconfig2.Args = []string{"sh", "-c", "cat && readlink /proc/self/ns/pid"}
	pconfig2.Stdin = stdinR2
	pconfig2.Stdout = &stdout2

	err = container.Run(&pconfig2)
	stdinR2.Close()
	defer stdinW2.Close()
	ok(t, err)

	pid2, err := pconfig2.Pid()
	ok(t, err)

	processes, err := container.Processes()
	ok(t, err)

	n := 0
	for i := range processes {
		if processes[i] == pid || processes[i] == pid2 {
			n++
		}
	}
	if n != 2 {
		t.Fatal("unexpected number of processes", processes, pid, pid2)
	}

	// Wait processes
	stdinW2.Close()
	waitProcess(&pconfig2, t)

	stdinW.Close()
	waitProcess(&pconfig, t)

	// Check that both processes live in the same pidns
	pidns := string(stdout.Bytes())
	ok(t, err)

	pidns2 := string(stdout2.Bytes())
	ok(t, err)

	if pidns != pidns2 {
		t.Fatal("The second process isn't in the required pid namespace", pidns, pidns2)
	}
}

func TestProcessEnv(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	ok(t, err)
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)

	container, err := factory.Create("test", config)
	ok(t, err)
	defer container.Destroy()

	var stdout bytes.Buffer
	pconfig := libcontainer.Process{
		Cwd:  "/",
		Args: []string{"sh", "-c", "env"},
		Env: []string{
			"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
			"HOSTNAME=integration",
			"TERM=xterm",
			"FOO=BAR",
		},
		Stdin:  nil,
		Stdout: &stdout,
	}
	err = container.Run(&pconfig)
	ok(t, err)

	// Wait for process
	waitProcess(&pconfig, t)

	outputEnv := string(stdout.Bytes())

	// Check that the environment has the key/value pair we added
	if !strings.Contains(outputEnv, "FOO=BAR") {
		t.Fatal("Environment doesn't have the expected FOO=BAR key/value pair: ", outputEnv)
	}

	// Make sure that HOME is set
	if !strings.Contains(outputEnv, "HOME=/root") {
		t.Fatal("Environment doesn't have HOME set: ", outputEnv)
	}
}

func TestProcessEmptyCaps(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	ok(t, err)
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	config.Capabilities = nil

	container, err := factory.Create("test", config)
	ok(t, err)
	defer container.Destroy()

	var stdout bytes.Buffer
	pconfig := libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"sh", "-c", "cat /proc/self/status"},
		Env:    standardEnvironment,
		Stdin:  nil,
		Stdout: &stdout,
	}
	err = container.Run(&pconfig)
	ok(t, err)

	// Wait for process
	waitProcess(&pconfig, t)

	outputStatus := string(stdout.Bytes())

	lines := strings.Split(outputStatus, "\n")

	effectiveCapsLine := ""
	for _, l := range lines {
		line := strings.TrimSpace(l)
		if strings.Contains(line, "CapEff:") {
			effectiveCapsLine = line
			break
		}
	}

	if effectiveCapsLine == "" {
		t.Fatal("Couldn't find effective caps: ", outputStatus)
	}
}

func TestProcessCaps(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	ok(t, err)
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)

	container, err := factory.Create("test", config)
	ok(t, err)
	defer container.Destroy()

	var stdout bytes.Buffer
	pconfig := libcontainer.Process{
		Cwd:          "/",
		Args:         []string{"sh", "-c", "cat /proc/self/status"},
		Env:          standardEnvironment,
		Stdin:        nil,
		Stdout:       &stdout,
		Capabilities: &configs.Capabilities{},
	}
	pconfig.Capabilities.Bounding = append(config.Capabilities.Bounding, "CAP_NET_ADMIN")
	pconfig.Capabilities.Permitted = append(config.Capabilities.Permitted, "CAP_NET_ADMIN")
	pconfig.Capabilities.Effective = append(config.Capabilities.Effective, "CAP_NET_ADMIN")
	pconfig.Capabilities.Inheritable = append(config.Capabilities.Inheritable, "CAP_NET_ADMIN")
	err = container.Run(&pconfig)
	ok(t, err)

	// Wait for process
	waitProcess(&pconfig, t)

	outputStatus := string(stdout.Bytes())

	lines := strings.Split(outputStatus, "\n")

	effectiveCapsLine := ""
	for _, l := range lines {
		line := strings.TrimSpace(l)
		if strings.Contains(line, "CapEff:") {
			effectiveCapsLine = line
			break
		}
	}

	if effectiveCapsLine == "" {
		t.Fatal("Couldn't find effective caps: ", outputStatus)
	}

	parts := strings.Split(effectiveCapsLine, ":")
	effectiveCapsStr := strings.TrimSpace(parts[1])

	effectiveCaps, err := strconv.ParseUint(effectiveCapsStr, 16, 64)
	if err != nil {
		t.Fatal("Could not parse effective caps", err)
	}

	var netAdminMask uint64
	var netAdminBit uint
	netAdminBit = 12 // from capability.h
	netAdminMask = 1 << netAdminBit
	if effectiveCaps&netAdminMask != netAdminMask {
		t.Fatal("CAP_NET_ADMIN is not set as expected")
	}
}

func TestAdditionalGroups(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	ok(t, err)
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)

	factory, err := libcontainer.New(root, libcontainer.Cgroupfs)
	ok(t, err)

	container, err := factory.Create("test", config)
	ok(t, err)
	defer container.Destroy()

	var stdout bytes.Buffer
	pconfig := libcontainer.Process{
		Cwd:              "/",
		Args:             []string{"sh", "-c", "id", "-Gn"},
		Env:              standardEnvironment,
		Stdin:            nil,
		Stdout:           &stdout,
		AdditionalGroups: []string{"plugdev", "audio"},
	}
	err = container.Run(&pconfig)
	ok(t, err)

	// Wait for process
	waitProcess(&pconfig, t)

	outputGroups := string(stdout.Bytes())

	// Check that the groups output has the groups that we specified
	if !strings.Contains(outputGroups, "audio") {
		t.Fatalf("Listed groups do not contain the audio group as expected: %v", outputGroups)
	}

	if !strings.Contains(outputGroups, "plugdev") {
		t.Fatalf("Listed groups do not contain the plugdev group as expected: %v", outputGroups)
	}
}

func TestFreeze(t *testing.T) {
	testFreeze(t, false)
}

func TestSystemdFreeze(t *testing.T) {
	if !systemd.UseSystemd() {
		t.Skip("Systemd is unsupported")
	}
	testFreeze(t, true)
}

func testFreeze(t *testing.T, systemd bool) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	ok(t, err)
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	f := factory
	if systemd {
		f = systemdFactory
	}

	container, err := f.Create("test", config)
	ok(t, err)
	defer container.Destroy()

	stdinR, stdinW, err := os.Pipe()
	ok(t, err)

	pconfig := &libcontainer.Process{
		Cwd:   "/",
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}
	err = container.Run(pconfig)
	stdinR.Close()
	defer stdinW.Close()
	ok(t, err)

	err = container.Pause()
	ok(t, err)
	state, err := container.Status()
	ok(t, err)
	err = container.Resume()
	ok(t, err)
	if state != libcontainer.Paused {
		t.Fatal("Unexpected state: ", state)
	}

	stdinW.Close()
	waitProcess(pconfig, t)
}

func TestCpuShares(t *testing.T) {
	testCpuShares(t, false)
}

func TestCpuSharesSystemd(t *testing.T) {
	if !systemd.UseSystemd() {
		t.Skip("Systemd is unsupported")
	}
	testCpuShares(t, true)
}

func testCpuShares(t *testing.T, systemd bool) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	if systemd {
		config.Cgroups.Parent = "system.slice"
	}
	config.Cgroups.Resources.CpuShares = 1

	_, _, err = runContainer(config, "", "ps")
	if err == nil {
		t.Fatalf("runContainer should failed with invalid CpuShares")
	}
}

func TestPids(t *testing.T) {
	testPids(t, false)
}

func TestPidsSystemd(t *testing.T) {
	if !systemd.UseSystemd() {
		t.Skip("Systemd is unsupported")
	}
	testPids(t, true)
}

func testPids(t *testing.T, systemd bool) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	if systemd {
		config.Cgroups.Parent = "system.slice"
	}
	config.Cgroups.Resources.PidsLimit = -1

	// Running multiple processes.
	_, ret, err := runContainer(config, "", "/bin/sh", "-c", "/bin/true | /bin/true | /bin/true | /bin/true")
	if err != nil && strings.Contains(err.Error(), "no such directory for pids.max") {
		t.Skip("PIDs cgroup is unsupported")
	}
	ok(t, err)

	if ret != 0 {
		t.Fatalf("expected fork() to succeed with no pids limit")
	}

	// Enforce a permissive limit. This needs to be fairly hand-wavey due to the
	// issues with running Go binaries with pids restrictions (see below).
	config.Cgroups.Resources.PidsLimit = 64
	_, ret, err = runContainer(config, "", "/bin/sh", "-c", `
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true`)
	if err != nil && strings.Contains(err.Error(), "no such directory for pids.max") {
		t.Skip("PIDs cgroup is unsupported")
	}
	ok(t, err)

	if ret != 0 {
		t.Fatalf("expected fork() to succeed with permissive pids limit")
	}

	// Enforce a restrictive limit. 64 * /bin/true + 1 * shell should cause this
	// to fail reliability.
	config.Cgroups.Resources.PidsLimit = 64
	out, _, err := runContainer(config, "", "/bin/sh", "-c", `
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true |
	/bin/true | /bin/true | /bin/true | /bin/true | /bin/true | /bin/true | bin/true | /bin/true`)
	if err != nil && strings.Contains(err.Error(), "no such directory for pids.max") {
		t.Skip("PIDs cgroup is unsupported")
	}
	if err != nil && !strings.Contains(out.String(), "sh: can't fork") {
		ok(t, err)
	}

	if err == nil {
		t.Fatalf("expected fork() to fail with restrictive pids limit")
	}

	// Minimal restrictions are not really supported, due to quirks in using Go
	// due to the fact that it spawns random processes. While we do our best with
	// late setting cgroup values, it's just too unreliable with very small pids.max.
	// As such, we don't test that case. YMMV.
}

func TestRunWithKernelMemory(t *testing.T) {
	testRunWithKernelMemory(t, false)
}

func TestRunWithKernelMemorySystemd(t *testing.T) {
	if !systemd.UseSystemd() {
		t.Skip("Systemd is unsupported")
	}
	testRunWithKernelMemory(t, true)
}

func testRunWithKernelMemory(t *testing.T, systemd bool) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	if systemd {
		config.Cgroups.Parent = "system.slice"
	}
	config.Cgroups.Resources.KernelMemory = 52428800

	_, _, err = runContainer(config, "", "ps")
	if err != nil {
		t.Fatalf("runContainer failed with kernel memory limit: %v", err)
	}
}

func TestContainerState(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	l, err := os.Readlink("/proc/1/ns/ipc")
	if err != nil {
		t.Fatal(err)
	}

	config := newTemplateConfig(rootfs)
	config.Namespaces = configs.Namespaces([]configs.Namespace{
		{Type: configs.NEWNS},
		{Type: configs.NEWUTS},
		// host for IPC
		//{Type: configs.NEWIPC},
		{Type: configs.NEWPID},
		{Type: configs.NEWNET},
	})

	container, err := factory.Create("test", config)
	if err != nil {
		t.Fatal(err)
	}
	defer container.Destroy()

	stdinR, stdinW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	p := &libcontainer.Process{
		Cwd:   "/",
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}
	err = container.Run(p)
	if err != nil {
		t.Fatal(err)
	}
	stdinR.Close()
	defer stdinW.Close()

	st, err := container.State()
	if err != nil {
		t.Fatal(err)
	}

	l1, err := os.Readlink(st.NamespacePaths[configs.NEWIPC])
	if err != nil {
		t.Fatal(err)
	}
	if l1 != l {
		t.Fatal("Container using non-host ipc namespace")
	}
	stdinW.Close()
	waitProcess(p, t)
}

func TestPassExtraFiles(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)

	container, err := factory.Create("test", config)
	if err != nil {
		t.Fatal(err)
	}
	defer container.Destroy()

	var stdout bytes.Buffer
	pipeout1, pipein1, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	pipeout2, pipein2, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	process := libcontainer.Process{
		Cwd:        "/",
		Args:       []string{"sh", "-c", "cd /proc/$$/fd; echo -n *; echo -n 1 >3; echo -n 2 >4"},
		Env:        []string{"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
		ExtraFiles: []*os.File{pipein1, pipein2},
		Stdin:      nil,
		Stdout:     &stdout,
	}
	err = container.Run(&process)
	if err != nil {
		t.Fatal(err)
	}

	waitProcess(&process, t)

	out := string(stdout.Bytes())
	// fd 5 is the directory handle for /proc/$$/fd
	if out != "0 1 2 3 4 5" {
		t.Fatalf("expected to have the file descriptors '0 1 2 3 4 5' passed to init, got '%s'", out)
	}
	var buf = []byte{0}
	_, err = pipeout1.Read(buf)
	if err != nil {
		t.Fatal(err)
	}
	out1 := string(buf)
	if out1 != "1" {
		t.Fatalf("expected first pipe to receive '1', got '%s'", out1)
	}

	_, err = pipeout2.Read(buf)
	if err != nil {
		t.Fatal(err)
	}
	out2 := string(buf)
	if out2 != "2" {
		t.Fatalf("expected second pipe to receive '2', got '%s'", out2)
	}
}

func TestMountCmds(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	tmpDir, err := ioutil.TempDir("", "tmpdir")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	config := newTemplateConfig(rootfs)
	config.Mounts = append(config.Mounts, &configs.Mount{
		Source:      tmpDir,
		Destination: "/tmp",
		Device:      "bind",
		Flags:       unix.MS_BIND | unix.MS_REC,
		PremountCmds: []configs.Command{
			{Path: "touch", Args: []string{filepath.Join(tmpDir, "hello")}},
			{Path: "touch", Args: []string{filepath.Join(tmpDir, "world")}},
		},
		PostmountCmds: []configs.Command{
			{Path: "cp", Args: []string{filepath.Join(rootfs, "tmp", "hello"), filepath.Join(rootfs, "tmp", "hello-backup")}},
			{Path: "cp", Args: []string{filepath.Join(rootfs, "tmp", "world"), filepath.Join(rootfs, "tmp", "world-backup")}},
		},
	})

	container, err := factory.Create("test", config)
	if err != nil {
		t.Fatal(err)
	}
	defer container.Destroy()

	pconfig := libcontainer.Process{
		Cwd:  "/",
		Args: []string{"sh", "-c", "env"},
		Env:  standardEnvironment,
	}
	err = container.Run(&pconfig)
	if err != nil {
		t.Fatal(err)
	}

	// Wait for process
	waitProcess(&pconfig, t)

	entries, err := ioutil.ReadDir(tmpDir)
	if err != nil {
		t.Fatal(err)
	}
	expected := []string{"hello", "hello-backup", "world", "world-backup"}
	for i, e := range entries {
		if e.Name() != expected[i] {
			t.Errorf("Got(%s), expect %s", e.Name(), expected[i])
		}
	}
}

func TestSysctl(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	ok(t, err)
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	config.Sysctl = map[string]string{
		"kernel.shmmni": "8192",
	}

	container, err := factory.Create("test", config)
	ok(t, err)
	defer container.Destroy()

	var stdout bytes.Buffer
	pconfig := libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"sh", "-c", "cat /proc/sys/kernel/shmmni"},
		Env:    standardEnvironment,
		Stdin:  nil,
		Stdout: &stdout,
	}
	err = container.Run(&pconfig)
	ok(t, err)

	// Wait for process
	waitProcess(&pconfig, t)

	shmmniOutput := strings.TrimSpace(string(stdout.Bytes()))
	if shmmniOutput != "8192" {
		t.Fatalf("kernel.shmmni property expected to be 8192, but is %s", shmmniOutput)
	}
}

func TestMountCgroupRO(t *testing.T) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)

	config.Mounts = append(config.Mounts, &configs.Mount{
		Destination: "/sys/fs/cgroup",
		Device:      "cgroup",
		Flags:       defaultMountFlags | unix.MS_RDONLY,
	})

	buffers, exitCode, err := runContainer(config, "", "mount")
	if err != nil {
		t.Fatalf("%s: %s", buffers, err)
	}
	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}
	mountInfo := buffers.Stdout.String()
	lines := strings.Split(mountInfo, "\n")
	for _, l := range lines {
		if strings.HasPrefix(l, "tmpfs on /sys/fs/cgroup") {
			if !strings.Contains(l, "ro") ||
				!strings.Contains(l, "nosuid") ||
				!strings.Contains(l, "nodev") ||
				!strings.Contains(l, "noexec") {
				t.Fatalf("Mode expected to contain 'ro,nosuid,nodev,noexec': %s", l)
			}
			if !strings.Contains(l, "mode=755") {
				t.Fatalf("Mode expected to contain 'mode=755': %s", l)
			}
			continue
		}
		if !strings.HasPrefix(l, "cgroup") {
			continue
		}
		if !strings.Contains(l, "ro") ||
			!strings.Contains(l, "nosuid") ||
			!strings.Contains(l, "nodev") ||
			!strings.Contains(l, "noexec") {
			t.Fatalf("Mode expected to contain 'ro,nosuid,nodev,noexec': %s", l)
		}
	}
}

func TestMountCgroupRW(t *testing.T) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)

	config.Mounts = append(config.Mounts, &configs.Mount{
		Destination: "/sys/fs/cgroup",
		Device:      "cgroup",
		Flags:       defaultMountFlags,
	})

	buffers, exitCode, err := runContainer(config, "", "mount")
	if err != nil {
		t.Fatalf("%s: %s", buffers, err)
	}
	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}
	mountInfo := buffers.Stdout.String()
	lines := strings.Split(mountInfo, "\n")
	for _, l := range lines {
		if strings.HasPrefix(l, "tmpfs on /sys/fs/cgroup") {
			if !strings.Contains(l, "rw") ||
				!strings.Contains(l, "nosuid") ||
				!strings.Contains(l, "nodev") ||
				!strings.Contains(l, "noexec") {
				t.Fatalf("Mode expected to contain 'rw,nosuid,nodev,noexec': %s", l)
			}
			if !strings.Contains(l, "mode=755") {
				t.Fatalf("Mode expected to contain 'mode=755': %s", l)
			}
			continue
		}
		if !strings.HasPrefix(l, "cgroup") {
			continue
		}
		if !strings.Contains(l, "rw") ||
			!strings.Contains(l, "nosuid") ||
			!strings.Contains(l, "nodev") ||
			!strings.Contains(l, "noexec") {
			t.Fatalf("Mode expected to contain 'rw,nosuid,nodev,noexec': %s", l)
		}
	}
}

func TestOomScoreAdj(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	ok(t, err)
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	config.OomScoreAdj = 200

	factory, err := libcontainer.New(root, libcontainer.Cgroupfs)
	ok(t, err)

	container, err := factory.Create("test", config)
	ok(t, err)
	defer container.Destroy()

	var stdout bytes.Buffer
	pconfig := libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"sh", "-c", "cat /proc/self/oom_score_adj"},
		Env:    standardEnvironment,
		Stdin:  nil,
		Stdout: &stdout,
	}
	err = container.Run(&pconfig)
	ok(t, err)

	// Wait for process
	waitProcess(&pconfig, t)
	outputOomScoreAdj := strings.TrimSpace(string(stdout.Bytes()))

	// Check that the oom_score_adj matches the value that was set as part of config.
	if outputOomScoreAdj != strconv.Itoa(config.OomScoreAdj) {
		t.Fatalf("Expected oom_score_adj %d; got %q", config.OomScoreAdj, outputOomScoreAdj)
	}
}

func TestHook(t *testing.T) {
	if testing.Short() {
		return
	}

	bundle, err := newTestBundle()
	ok(t, err)
	defer remove(bundle)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	expectedBundle := bundle
	config.Labels = append(config.Labels, fmt.Sprintf("bundle=%s", expectedBundle))

	getRootfsFromBundle := func(bundle string) (string, error) {
		f, err := os.Open(filepath.Join(bundle, "config.json"))
		if err != nil {
			return "", err
		}

		var config configs.Config
		if err = json.NewDecoder(f).Decode(&config); err != nil {
			return "", err
		}
		return config.Rootfs, nil
	}

	config.Hooks = &configs.Hooks{
		Prestart: []configs.Hook{
			configs.NewFunctionHook(func(s configs.HookState) error {
				if s.Bundle != expectedBundle {
					t.Fatalf("Expected prestart hook bundlePath '%s'; got '%s'", expectedBundle, s.Bundle)
				}

				root, err := getRootfsFromBundle(s.Bundle)
				if err != nil {
					return err
				}
				f, err := os.Create(filepath.Join(root, "test"))
				if err != nil {
					return err
				}
				return f.Close()
			}),
		},
		Poststart: []configs.Hook{
			configs.NewFunctionHook(func(s configs.HookState) error {
				if s.Bundle != expectedBundle {
					t.Fatalf("Expected poststart hook bundlePath '%s'; got '%s'", expectedBundle, s.Bundle)
				}

				root, err := getRootfsFromBundle(s.Bundle)
				if err != nil {
					return err
				}
				return ioutil.WriteFile(filepath.Join(root, "test"), []byte("hello world"), 0755)
			}),
		},
		Poststop: []configs.Hook{
			configs.NewFunctionHook(func(s configs.HookState) error {
				if s.Bundle != expectedBundle {
					t.Fatalf("Expected poststop hook bundlePath '%s'; got '%s'", expectedBundle, s.Bundle)
				}

				root, err := getRootfsFromBundle(s.Bundle)
				if err != nil {
					return err
				}
				return os.RemoveAll(filepath.Join(root, "test"))
			}),
		},
	}

	// write config of json format into config.json under bundle
	f, err := os.OpenFile(filepath.Join(bundle, "config.json"), os.O_CREATE|os.O_RDWR, 0644)
	ok(t, err)
	ok(t, json.NewEncoder(f).Encode(config))

	container, err := factory.Create("test", config)
	ok(t, err)

	var stdout bytes.Buffer
	pconfig := libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"sh", "-c", "ls /test"},
		Env:    standardEnvironment,
		Stdin:  nil,
		Stdout: &stdout,
	}
	err = container.Run(&pconfig)
	ok(t, err)

	// Wait for process
	waitProcess(&pconfig, t)

	outputLs := string(stdout.Bytes())

	// Check that the ls output has the expected file touched by the prestart hook
	if !strings.Contains(outputLs, "/test") {
		container.Destroy()
		t.Fatalf("ls output doesn't have the expected file: %s", outputLs)
	}

	// Check that the file is written by the poststart hook
	testFilePath := filepath.Join(rootfs, "test")
	contents, err := ioutil.ReadFile(testFilePath)
	if err != nil {
		t.Fatalf("cannot read file '%s': %s", testFilePath, err)
	}
	if string(contents) != "hello world" {
		t.Fatalf("Expected test file to contain 'hello world'; got '%s'", string(contents))
	}

	if err := container.Destroy(); err != nil {
		t.Fatalf("container destroy %s", err)
	}
	fi, err := os.Stat(filepath.Join(rootfs, "test"))
	if err == nil || !os.IsNotExist(err) {
		t.Fatalf("expected file to not exist, got %s", fi.Name())
	}
}

func TestSTDIOPermissions(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)
	buffers, exitCode, err := runContainer(config, "", "sh", "-c", "echo hi > /dev/stderr")
	ok(t, err)
	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}

	if actual := strings.Trim(buffers.Stderr.String(), "\n"); actual != "hi" {
		t.Fatalf("stderr should equal be equal %q %q", actual, "hi")
	}
}

func unmountOp(path string) error {
	if err := unix.Unmount(path, unix.MNT_DETACH); err != nil {
		return err
	}
	return nil
}

// Launch container with rootfsPropagation in rslave mode. Also
// bind mount a volume /mnt1host at /mnt1cont at the time of launch. Now do
// another mount on host (/mnt1host/mnt2host) and this new mount should
// propagate to container (/mnt1cont/mnt2host)
func TestRootfsPropagationSlaveMount(t *testing.T) {
	var mountPropagated bool
	var dir1cont string
	var dir2cont string

	dir1cont = "/root/mnt1cont"

	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)

	config.RootPropagation = unix.MS_SLAVE | unix.MS_REC

	// Bind mount a volume
	dir1host, err := ioutil.TempDir("", "mnt1host")
	ok(t, err)
	defer os.RemoveAll(dir1host)

	// Make this dir a "shared" mount point. This will make sure a
	// slave relationship can be established in container.
	err = unix.Mount(dir1host, dir1host, "bind", unix.MS_BIND|unix.MS_REC, "")
	ok(t, err)
	err = unix.Mount("", dir1host, "", unix.MS_SHARED|unix.MS_REC, "")
	ok(t, err)
	defer unmountOp(dir1host)

	config.Mounts = append(config.Mounts, &configs.Mount{
		Source:      dir1host,
		Destination: dir1cont,
		Device:      "bind",
		Flags:       unix.MS_BIND | unix.MS_REC})

	// TODO: systemd specific processing
	f := factory

	container, err := f.Create("testSlaveMount", config)
	ok(t, err)
	defer container.Destroy()

	stdinR, stdinW, err := os.Pipe()
	ok(t, err)

	pconfig := &libcontainer.Process{
		Cwd:   "/",
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}

	err = container.Run(pconfig)
	stdinR.Close()
	defer stdinW.Close()
	ok(t, err)

	// Create mnt1host/mnt2host and bind mount itself on top of it. This
	// should be visible in container.
	dir2host, err := ioutil.TempDir(dir1host, "mnt2host")
	ok(t, err)
	defer os.RemoveAll(dir2host)

	err = unix.Mount(dir2host, dir2host, "bind", unix.MS_BIND, "")
	defer unmountOp(dir2host)
	ok(t, err)

	// Run "cat /proc/self/mountinfo" in container and look at mount points.
	var stdout2 bytes.Buffer

	stdinR2, stdinW2, err := os.Pipe()
	ok(t, err)

	pconfig2 := &libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"cat", "/proc/self/mountinfo"},
		Env:    standardEnvironment,
		Stdin:  stdinR2,
		Stdout: &stdout2,
	}

	err = container.Run(pconfig2)
	stdinR2.Close()
	defer stdinW2.Close()
	ok(t, err)

	stdinW2.Close()
	waitProcess(pconfig2, t)
	stdinW.Close()
	waitProcess(pconfig, t)

	mountPropagated = false
	dir2cont = filepath.Join(dir1cont, filepath.Base(dir2host))

	propagationInfo := string(stdout2.Bytes())
	lines := strings.Split(propagationInfo, "\n")
	for _, l := range lines {
		linefields := strings.Split(l, " ")
		if len(linefields) < 5 {
			continue
		}

		if linefields[4] == dir2cont {
			mountPropagated = true
			break
		}
	}

	if mountPropagated != true {
		t.Fatalf("Mount on host %s did not propagate in container at %s\n", dir2host, dir2cont)
	}
}

// Launch container with rootfsPropagation 0 so no propagation flags are
// applied. Also bind mount a volume /mnt1host at /mnt1cont at the time of
// launch. Now do a mount in container (/mnt1cont/mnt2cont) and this new
// mount should propagate to host (/mnt1host/mnt2cont)

func TestRootfsPropagationSharedMount(t *testing.T) {
	var dir1cont string
	var dir2cont string

	dir1cont = "/root/mnt1cont"

	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)
	config.RootPropagation = unix.MS_PRIVATE

	// Bind mount a volume
	dir1host, err := ioutil.TempDir("", "mnt1host")
	ok(t, err)
	defer os.RemoveAll(dir1host)

	// Make this dir a "shared" mount point. This will make sure a
	// shared relationship can be established in container.
	err = unix.Mount(dir1host, dir1host, "bind", unix.MS_BIND|unix.MS_REC, "")
	ok(t, err)
	err = unix.Mount("", dir1host, "", unix.MS_SHARED|unix.MS_REC, "")
	ok(t, err)
	defer unmountOp(dir1host)

	config.Mounts = append(config.Mounts, &configs.Mount{
		Source:      dir1host,
		Destination: dir1cont,
		Device:      "bind",
		Flags:       unix.MS_BIND | unix.MS_REC})

	// TODO: systemd specific processing
	f := factory

	container, err := f.Create("testSharedMount", config)
	ok(t, err)
	defer container.Destroy()

	stdinR, stdinW, err := os.Pipe()
	ok(t, err)

	pconfig := &libcontainer.Process{
		Cwd:   "/",
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}

	err = container.Run(pconfig)
	stdinR.Close()
	defer stdinW.Close()
	ok(t, err)

	// Create mnt1host/mnt2cont.  This will become visible inside container
	// at mnt1cont/mnt2cont. Bind mount itself on top of it. This
	// should be visible on host now.
	dir2host, err := ioutil.TempDir(dir1host, "mnt2cont")
	ok(t, err)
	defer os.RemoveAll(dir2host)

	dir2cont = filepath.Join(dir1cont, filepath.Base(dir2host))

	// Mount something in container and see if it is visible on host.
	var stdout2 bytes.Buffer

	stdinR2, stdinW2, err := os.Pipe()
	ok(t, err)

	pconfig2 := &libcontainer.Process{
		Cwd:          "/",
		Args:         []string{"mount", "--bind", dir2cont, dir2cont},
		Env:          standardEnvironment,
		Stdin:        stdinR2,
		Stdout:       &stdout2,
		Capabilities: &configs.Capabilities{},
	}

	// Provide CAP_SYS_ADMIN
	pconfig2.Capabilities.Bounding = append(config.Capabilities.Bounding, "CAP_SYS_ADMIN")
	pconfig2.Capabilities.Permitted = append(config.Capabilities.Permitted, "CAP_SYS_ADMIN")
	pconfig2.Capabilities.Effective = append(config.Capabilities.Effective, "CAP_SYS_ADMIN")
	pconfig2.Capabilities.Inheritable = append(config.Capabilities.Inheritable, "CAP_SYS_ADMIN")

	err = container.Run(pconfig2)
	stdinR2.Close()
	defer stdinW2.Close()
	ok(t, err)

	// Wait for process
	stdinW2.Close()
	waitProcess(pconfig2, t)
	stdinW.Close()
	waitProcess(pconfig, t)

	defer unmountOp(dir2host)

	// Check if mount is visible on host or not.
	out, err := exec.Command("findmnt", "-n", "-f", "-oTARGET", dir2host).CombinedOutput()
	outtrim := strings.TrimSpace(string(out))
	if err != nil {
		t.Logf("findmnt error %q: %q", err, outtrim)
	}

	if string(outtrim) != dir2host {
		t.Fatalf("Mount in container on %s did not propagate to host on %s. finmnt output=%s", dir2cont, dir2host, outtrim)
	}
}

func TestPIDHost(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	l, err := os.Readlink("/proc/1/ns/pid")
	ok(t, err)

	config := newTemplateConfig(rootfs)
	config.Namespaces.Remove(configs.NEWPID)
	buffers, exitCode, err := runContainer(config, "", "readlink", "/proc/self/ns/pid")
	ok(t, err)

	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}

	if actual := strings.Trim(buffers.Stdout.String(), "\n"); actual != l {
		t.Fatalf("ipc link not equal to host link %q %q", actual, l)
	}
}

func TestInitJoinPID(t *testing.T) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	// Execute a long-running container
	container1, err := newContainer(newTemplateConfig(rootfs))
	ok(t, err)
	defer container1.Destroy()

	stdinR1, stdinW1, err := os.Pipe()
	ok(t, err)
	init1 := &libcontainer.Process{
		Cwd:   "/",
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR1,
	}
	err = container1.Run(init1)
	stdinR1.Close()
	defer stdinW1.Close()
	ok(t, err)

	// get the state of the first container
	state1, err := container1.State()
	ok(t, err)
	pidns1 := state1.NamespacePaths[configs.NEWPID]

	// Run a container inside the existing pidns but with different cgroups
	config2 := newTemplateConfig(rootfs)
	config2.Namespaces.Add(configs.NEWPID, pidns1)
	config2.Cgroups.Path = "integration/test2"
	container2, err := newContainerWithName("testCT2", config2)
	ok(t, err)
	defer container2.Destroy()

	stdinR2, stdinW2, err := os.Pipe()
	ok(t, err)
	init2 := &libcontainer.Process{
		Cwd:   "/",
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR2,
	}
	err = container2.Run(init2)
	stdinR2.Close()
	defer stdinW2.Close()
	ok(t, err)
	// get the state of the second container
	state2, err := container2.State()
	ok(t, err)

	ns1, err := os.Readlink(fmt.Sprintf("/proc/%d/ns/pid", state1.InitProcessPid))
	ok(t, err)
	ns2, err := os.Readlink(fmt.Sprintf("/proc/%d/ns/pid", state2.InitProcessPid))
	ok(t, err)
	if ns1 != ns2 {
		t.Errorf("pidns(%s), wanted %s", ns2, ns1)
	}

	// check that namespaces are not the same
	if reflect.DeepEqual(state2.NamespacePaths, state1.NamespacePaths) {
		t.Errorf("Namespaces(%v), original %v", state2.NamespacePaths,
			state1.NamespacePaths)
	}
	// check that pidns is joined correctly. The initial container process list
	// should contain the second container's init process
	buffers := newStdBuffers()
	ps := &libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"ps"},
		Env:    standardEnvironment,
		Stdout: buffers.Stdout,
	}
	err = container1.Run(ps)
	ok(t, err)
	waitProcess(ps, t)

	// Stop init processes one by one. Stop the second container should
	// not stop the first.
	stdinW2.Close()
	waitProcess(init2, t)
	stdinW1.Close()
	waitProcess(init1, t)

	out := strings.TrimSpace(buffers.Stdout.String())
	// output of ps inside the initial PID namespace should have
	// 1 line of header,
	// 2 lines of init processes,
	// 1 line of ps process
	if len(strings.Split(out, "\n")) != 4 {
		t.Errorf("unexpected running process, output %q", out)
	}
}

func TestInitJoinNetworkAndUser(t *testing.T) {
	if _, err := os.Stat("/proc/self/ns/user"); os.IsNotExist(err) {
		t.Skip("userns is unsupported")
	}
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	// Execute a long-running container
	config1 := newTemplateConfig(rootfs)
	config1.UidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
	config1.GidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
	config1.Namespaces = append(config1.Namespaces, configs.Namespace{Type: configs.NEWUSER})
	container1, err := newContainer(config1)
	ok(t, err)
	defer container1.Destroy()

	stdinR1, stdinW1, err := os.Pipe()
	ok(t, err)
	init1 := &libcontainer.Process{
		Cwd:   "/",
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR1,
	}
	err = container1.Run(init1)
	stdinR1.Close()
	defer stdinW1.Close()
	ok(t, err)

	// get the state of the first container
	state1, err := container1.State()
	ok(t, err)
	netns1 := state1.NamespacePaths[configs.NEWNET]
	userns1 := state1.NamespacePaths[configs.NEWUSER]

	// Run a container inside the existing pidns but with different cgroups
	rootfs2, err := newRootfs()
	ok(t, err)
	defer remove(rootfs2)

	config2 := newTemplateConfig(rootfs2)
	config2.UidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
	config2.GidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
	config2.Namespaces.Add(configs.NEWNET, netns1)
	config2.Namespaces.Add(configs.NEWUSER, userns1)
	config2.Cgroups.Path = "integration/test2"
	container2, err := newContainerWithName("testCT2", config2)
	ok(t, err)
	defer container2.Destroy()

	stdinR2, stdinW2, err := os.Pipe()
	ok(t, err)
	init2 := &libcontainer.Process{
		Cwd:   "/",
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR2,
	}
	err = container2.Run(init2)
	stdinR2.Close()
	defer stdinW2.Close()
	ok(t, err)

	// get the state of the second container
	state2, err := container2.State()
	ok(t, err)

	for _, ns := range []string{"net", "user"} {
		ns1, err := os.Readlink(fmt.Sprintf("/proc/%d/ns/%s", state1.InitProcessPid, ns))
		ok(t, err)
		ns2, err := os.Readlink(fmt.Sprintf("/proc/%d/ns/%s", state2.InitProcessPid, ns))
		ok(t, err)
		if ns1 != ns2 {
			t.Errorf("%s(%s), wanted %s", ns, ns2, ns1)
		}
	}

	// check that namespaces are not the same
	if reflect.DeepEqual(state2.NamespacePaths, state1.NamespacePaths) {
		t.Errorf("Namespaces(%v), original %v", state2.NamespacePaths,
			state1.NamespacePaths)
	}
	// Stop init processes one by one. Stop the second container should
	// not stop the first.
	stdinW2.Close()
	waitProcess(init2, t)
	stdinW1.Close()
	waitProcess(init1, t)
}

func TestTmpfsCopyUp(t *testing.T) {
	if testing.Short() {
		return
	}
	root, err := newTestRoot()
	ok(t, err)
	defer os.RemoveAll(root)

	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)

	config.Mounts = append(config.Mounts, &configs.Mount{
		Source:      "tmpfs",
		Destination: "/etc",
		Device:      "tmpfs",
		Extensions:  configs.EXT_COPYUP,
	})

	factory, err := libcontainer.New(root, libcontainer.Cgroupfs)
	ok(t, err)

	container, err := factory.Create("test", config)
	ok(t, err)
	defer container.Destroy()

	var stdout bytes.Buffer
	pconfig := libcontainer.Process{
		Args:   []string{"ls", "/etc/passwd"},
		Env:    standardEnvironment,
		Stdin:  nil,
		Stdout: &stdout,
	}
	err = container.Run(&pconfig)
	ok(t, err)

	// Wait for process
	waitProcess(&pconfig, t)

	outputLs := string(stdout.Bytes())

	// Check that the ls output has /etc/passwd
	if !strings.Contains(outputLs, "/etc/passwd") {
		t.Fatalf("/etc/passwd not copied up as expected: %v", outputLs)
	}
}
