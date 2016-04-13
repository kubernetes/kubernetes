package integration

import (
	"bufio"
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/configs"
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

func TestCheckpoint(t *testing.T) {
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
		Flags:       defaultMountFlags | syscall.MS_RDONLY,
	})

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

	err = container.Start(&pconfig)
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

	imagesDir, err := ioutil.TempDir("", "criu")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(imagesDir)

	checkpointOpts := &libcontainer.CriuOpts{
		ImagesDirectory: imagesDir,
		WorkDirectory:   imagesDir,
	}
	dumpLog := filepath.Join(checkpointOpts.WorkDirectory, "dump.log")
	restoreLog := filepath.Join(checkpointOpts.WorkDirectory, "restore.log")

	if err := container.Checkpoint(checkpointOpts); err != nil {
		showFile(t, dumpLog)
		t.Fatal(err)
	}

	state, err := container.Status()
	if err != nil {
		t.Fatal(err)
	}

	if state != libcontainer.Running {
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
