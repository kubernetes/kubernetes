package integration

import (
	"bufio"
	"bytes"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/configs"

	"golang.org/x/sys/unix"
)

func showFile(t *testing.T, fname string) error {
	t.Logf("=== %s ===\n", fname)

	f, err := os.Open(fname)
	if err != nil {
		t.Log(err)
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		t.Log(scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	t.Logf("=== END ===\n")

	return nil
}

func TestUsernsCheckpoint(t *testing.T) {
	if _, err := os.Stat("/proc/self/ns/user"); os.IsNotExist(err) {
		t.Skip("userns is unsupported")
	}
	cmd := exec.Command("criu", "check", "--feature", "userns")
	if err := cmd.Run(); err != nil {
		t.Skip("Unable to c/r a container with userns")
	}
	testCheckpoint(t, true)
}

func TestCheckpoint(t *testing.T) {
	testCheckpoint(t, false)
}

func testCheckpoint(t *testing.T, userns bool) {
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

	config := newTemplateConfig(rootfs)

	config.Mounts = append(config.Mounts, &configs.Mount{
		Destination: "/sys/fs/cgroup",
		Device:      "cgroup",
		Flags:       defaultMountFlags | unix.MS_RDONLY,
	})

	if userns {
		config.UidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
		config.GidMappings = []configs.IDMap{{HostID: 0, ContainerID: 0, Size: 1000}}
		config.Namespaces = append(config.Namespaces, configs.Namespace{Type: configs.NEWUSER})
	}

	factory, err := libcontainer.New(root, libcontainer.Cgroupfs)

	if err != nil {
		t.Fatal(err)
	}

	container, err := factory.Create("test", config)
	if err != nil {
		t.Fatal(err)
	}
	defer container.Destroy()

	stdinR, stdinW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}

	var stdout bytes.Buffer

	pconfig := libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"cat"},
		Env:    standardEnvironment,
		Stdin:  stdinR,
		Stdout: &stdout,
	}

	err = container.Run(&pconfig)
	stdinR.Close()
	defer stdinW.Close()
	if err != nil {
		t.Fatal(err)
	}

	pid, err := pconfig.Pid()
	if err != nil {
		t.Fatal(err)
	}

	process, err := os.FindProcess(pid)
	if err != nil {
		t.Fatal(err)
	}

	parentDir, err := ioutil.TempDir("", "criu-parent")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(parentDir)

	preDumpOpts := &libcontainer.CriuOpts{
		ImagesDirectory: parentDir,
		WorkDirectory:   parentDir,
		PreDump:         true,
	}
	preDumpLog := filepath.Join(preDumpOpts.WorkDirectory, "dump.log")

	if err := container.Checkpoint(preDumpOpts); err != nil {
		showFile(t, preDumpLog)
		t.Fatal(err)
	}

	state, err := container.Status()
	if err != nil {
		t.Fatal(err)
	}

	if state != libcontainer.Running {
		t.Fatal("Unexpected preDump state: ", state)
	}

	imagesDir, err := ioutil.TempDir("", "criu")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(imagesDir)

	checkpointOpts := &libcontainer.CriuOpts{
		ImagesDirectory: imagesDir,
		WorkDirectory:   imagesDir,
		ParentImage:     "../criu-parent",
	}
	dumpLog := filepath.Join(checkpointOpts.WorkDirectory, "dump.log")
	restoreLog := filepath.Join(checkpointOpts.WorkDirectory, "restore.log")

	if err := container.Checkpoint(checkpointOpts); err != nil {
		showFile(t, dumpLog)
		t.Fatal(err)
	}

	state, err = container.Status()
	if err != nil {
		t.Fatal(err)
	}

	if state != libcontainer.Stopped {
		t.Fatal("Unexpected state checkpoint: ", state)
	}

	stdinW.Close()
	_, err = process.Wait()
	if err != nil {
		t.Fatal(err)
	}

	// reload the container
	container, err = factory.Load("test")
	if err != nil {
		t.Fatal(err)
	}

	restoreStdinR, restoreStdinW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}

	restoreProcessConfig := &libcontainer.Process{
		Cwd:    "/",
		Stdin:  restoreStdinR,
		Stdout: &stdout,
	}

	err = container.Restore(restoreProcessConfig, checkpointOpts)
	restoreStdinR.Close()
	defer restoreStdinW.Close()
	if err != nil {
		showFile(t, restoreLog)
		t.Fatal(err)
	}

	state, err = container.Status()
	if err != nil {
		t.Fatal(err)
	}
	if state != libcontainer.Running {
		t.Fatal("Unexpected restore state: ", state)
	}

	pid, err = restoreProcessConfig.Pid()
	if err != nil {
		t.Fatal(err)
	}

	process, err = os.FindProcess(pid)
	if err != nil {
		t.Fatal(err)
	}

	_, err = restoreStdinW.WriteString("Hello!")
	if err != nil {
		t.Fatal(err)
	}

	restoreStdinW.Close()
	s, err := process.Wait()
	if err != nil {
		t.Fatal(err)
	}

	if !s.Success() {
		t.Fatal(s.String(), pid)
	}

	output := string(stdout.Bytes())
	if !strings.Contains(output, "Hello!") {
		t.Fatal("Did not restore the pipe correctly:", output)
	}
}
