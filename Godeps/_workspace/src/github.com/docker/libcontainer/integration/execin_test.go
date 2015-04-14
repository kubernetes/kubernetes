package integration

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/docker/libcontainer"
)

func TestExecIn(t *testing.T) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)
	container, err := newContainer(config)
	ok(t, err)
	defer container.Destroy()

	// Execute a first process in the container
	stdinR, stdinW, err := os.Pipe()
	ok(t, err)
	process := &libcontainer.Process{
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}
	err = container.Start(process)
	stdinR.Close()
	defer stdinW.Close()
	ok(t, err)

	buffers := newStdBuffers()
	ps := &libcontainer.Process{
		Args:   []string{"ps"},
		Env:    standardEnvironment,
		Stdin:  buffers.Stdin,
		Stdout: buffers.Stdout,
		Stderr: buffers.Stderr,
	}
	err = container.Start(ps)
	ok(t, err)
	_, err = ps.Wait()
	ok(t, err)
	stdinW.Close()
	if _, err := process.Wait(); err != nil {
		t.Log(err)
	}
	out := buffers.Stdout.String()
	if !strings.Contains(out, "cat") || !strings.Contains(out, "ps") {
		t.Fatalf("unexpected running process, output %q", out)
	}
}

func TestExecInRlimit(t *testing.T) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)
	container, err := newContainer(config)
	ok(t, err)
	defer container.Destroy()

	stdinR, stdinW, err := os.Pipe()
	ok(t, err)
	process := &libcontainer.Process{
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}
	err = container.Start(process)
	stdinR.Close()
	defer stdinW.Close()
	ok(t, err)

	buffers := newStdBuffers()
	ps := &libcontainer.Process{
		Args:   []string{"/bin/sh", "-c", "ulimit -n"},
		Env:    standardEnvironment,
		Stdin:  buffers.Stdin,
		Stdout: buffers.Stdout,
		Stderr: buffers.Stderr,
	}
	err = container.Start(ps)
	ok(t, err)
	_, err = ps.Wait()
	ok(t, err)
	stdinW.Close()
	if _, err := process.Wait(); err != nil {
		t.Log(err)
	}
	out := buffers.Stdout.String()
	if limit := strings.TrimSpace(out); limit != "1025" {
		t.Fatalf("expected rlimit to be 1025, got %s", limit)
	}
}

func TestExecInError(t *testing.T) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)
	container, err := newContainer(config)
	ok(t, err)
	defer container.Destroy()

	// Execute a first process in the container
	stdinR, stdinW, err := os.Pipe()
	ok(t, err)
	process := &libcontainer.Process{
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}
	err = container.Start(process)
	stdinR.Close()
	defer func() {
		stdinW.Close()
		if _, err := process.Wait(); err != nil {
			t.Log(err)
		}
	}()
	ok(t, err)

	unexistent := &libcontainer.Process{
		Args: []string{"unexistent"},
		Env:  standardEnvironment,
	}
	err = container.Start(unexistent)
	if err == nil {
		t.Fatal("Should be an error")
	}
	if !strings.Contains(err.Error(), "executable file not found") {
		t.Fatalf("Should be error about not found executable, got %s", err)
	}
}

func TestExecInTTY(t *testing.T) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)
	container, err := newContainer(config)
	ok(t, err)
	defer container.Destroy()

	// Execute a first process in the container
	stdinR, stdinW, err := os.Pipe()
	ok(t, err)
	process := &libcontainer.Process{
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}
	err = container.Start(process)
	stdinR.Close()
	defer stdinW.Close()
	ok(t, err)

	var stdout bytes.Buffer
	ps := &libcontainer.Process{
		Args: []string{"ps"},
		Env:  standardEnvironment,
	}
	console, err := ps.NewConsole(0)
	copy := make(chan struct{})
	go func() {
		io.Copy(&stdout, console)
		close(copy)
	}()
	ok(t, err)
	err = container.Start(ps)
	ok(t, err)
	select {
	case <-time.After(5 * time.Second):
		t.Fatal("Waiting for copy timed out")
	case <-copy:
	}
	_, err = ps.Wait()
	ok(t, err)
	stdinW.Close()
	if _, err := process.Wait(); err != nil {
		t.Log(err)
	}
	out := stdout.String()
	if !strings.Contains(out, "cat") || !strings.Contains(string(out), "ps") {
		t.Fatalf("unexpected running process, output %q", out)
	}
}

func TestExecInEnvironment(t *testing.T) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	ok(t, err)
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)
	container, err := newContainer(config)
	ok(t, err)
	defer container.Destroy()

	// Execute a first process in the container
	stdinR, stdinW, err := os.Pipe()
	ok(t, err)
	process := &libcontainer.Process{
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}
	err = container.Start(process)
	stdinR.Close()
	defer stdinW.Close()
	ok(t, err)

	buffers := newStdBuffers()
	process2 := &libcontainer.Process{
		Args: []string{"env"},
		Env: []string{
			"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
			"DEBUG=true",
			"DEBUG=false",
			"ENV=test",
		},
		Stdin:  buffers.Stdin,
		Stdout: buffers.Stdout,
		Stderr: buffers.Stderr,
	}
	err = container.Start(process2)
	ok(t, err)
	if _, err := process2.Wait(); err != nil {
		out := buffers.Stdout.String()
		t.Fatal(err, out)
	}
	stdinW.Close()
	if _, err := process.Wait(); err != nil {
		t.Log(err)
	}
	out := buffers.Stdout.String()
	// check execin's process environment
	if !strings.Contains(out, "DEBUG=false") ||
		!strings.Contains(out, "ENV=test") ||
		!strings.Contains(out, "HOME=/root") ||
		!strings.Contains(out, "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin") ||
		strings.Contains(out, "DEBUG=true") {
		t.Fatalf("unexpected running process, output %q", out)
	}
}

func TestExecinPassExtraFiles(t *testing.T) {
	if testing.Short() {
		return
	}
	rootfs, err := newRootfs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)
	config := newTemplateConfig(rootfs)
	container, err := newContainer(config)
	if err != nil {
		t.Fatal(err)
	}
	defer container.Destroy()

	// Execute a first process in the container
	stdinR, stdinW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	process := &libcontainer.Process{
		Args:  []string{"cat"},
		Env:   standardEnvironment,
		Stdin: stdinR,
	}
	err = container.Start(process)
	stdinR.Close()
	defer stdinW.Close()
	if err != nil {
		t.Fatal(err)
	}

	var stdout bytes.Buffer
	pipeout1, pipein1, err := os.Pipe()
	pipeout2, pipein2, err := os.Pipe()
	inprocess := &libcontainer.Process{
		Args:       []string{"sh", "-c", "cd /proc/$$/fd; echo -n *; echo -n 1 >3; echo -n 2 >4"},
		Env:        []string{"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
		ExtraFiles: []*os.File{pipein1, pipein2},
		Stdin:      nil,
		Stdout:     &stdout,
	}
	err = container.Start(inprocess)
	if err != nil {
		t.Fatal(err)
	}

	waitProcess(inprocess, t)

	out := string(stdout.Bytes())
	// fd 5 is the directory handle for /proc/$$/fd
	if out != "0 1 2 3 4 5" {
		t.Fatalf("expected to have the file descriptors '0 1 2 3 4 5' passed to exec, got '%s'", out)
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
