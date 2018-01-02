// +build linux

package containerd

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/containerd/cgroups"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/linux/runcopts"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func TestTaskUpdate(t *testing.T) {
	t.Parallel()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err := client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}
	limit := int64(32 * 1024 * 1024)
	memory := func(_ context.Context, _ *Client, _ *containers.Container, s *specs.Spec) error {
		s.Linux.Resources.Memory = &specs.LinuxMemory{
			Limit: &limit,
		}
		return nil
	}
	container, err := client.NewContainer(ctx, id, WithNewSpec(WithImageConfig(image), withProcessArgs("sleep", "30"), memory), WithNewSnapshot(id, image))
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	task, err := container.NewTask(ctx, empty())
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	// check that the task has a limit of 32mb
	cgroup, err := cgroups.Load(cgroups.V1, cgroups.PidPath(int(task.Pid())))
	if err != nil {
		t.Error(err)
		return
	}
	stat, err := cgroup.Stat(cgroups.IgnoreNotExist)
	if err != nil {
		t.Error(err)
		return
	}
	if int64(stat.Memory.Usage.Limit) != limit {
		t.Errorf("expected memory limit to be set to %d but received %d", limit, stat.Memory.Usage.Limit)
		return
	}
	limit = 64 * 1024 * 1024
	if err := task.Update(ctx, WithResources(&specs.LinuxResources{
		Memory: &specs.LinuxMemory{
			Limit: &limit,
		},
	})); err != nil {
		t.Error(err)
	}
	// check that the task has a limit of 64mb
	if stat, err = cgroup.Stat(cgroups.IgnoreNotExist); err != nil {
		t.Error(err)
		return
	}
	if int64(stat.Memory.Usage.Limit) != limit {
		t.Errorf("expected memory limit to be set to %d but received %d", limit, stat.Memory.Usage.Limit)
	}
	if err := task.Kill(ctx, unix.SIGKILL); err != nil {
		t.Error(err)
		return
	}

	<-statusC
}

func TestShimInCgroup(t *testing.T) {
	t.Parallel()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err := client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}
	container, err := client.NewContainer(ctx, id, WithNewSpec(WithImageConfig(image), WithProcessArgs("sleep", "30")), WithNewSnapshot(id, image))
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)
	// create a cgroup for the shim to use
	path := "/containerd/shim"
	cg, err := cgroups.New(cgroups.V1, cgroups.StaticPath(path), &specs.LinuxResources{})
	if err != nil {
		t.Error(err)
		return
	}
	defer cg.Delete()

	task, err := container.NewTask(ctx, empty(), func(_ context.Context, client *Client, r *TaskInfo) error {
		r.Options = &runcopts.CreateOptions{
			ShimCgroup: path,
		}
		return nil
	})
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	// check to see if the shim is inside the cgroup
	processes, err := cg.Processes(cgroups.Devices, false)
	if err != nil {
		t.Error(err)
		return
	}
	if len(processes) == 0 {
		t.Errorf("created cgroup should have atleast one process inside: %d", len(processes))
	}
	if err := task.Kill(ctx, unix.SIGKILL); err != nil {
		t.Error(err)
		return
	}

	<-statusC
}

func TestDaemonRestart(t *testing.T) {
	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}

	container, err := client.NewContainer(ctx, id, WithNewSpec(withImageConfig(image), withProcessArgs("sleep", "30")), withNewSnapshot(id, image))
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	task, err := container.NewTask(ctx, empty())
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	if err := task.Start(ctx); err != nil {
		t.Error(err)
		return
	}

	var exitStatus ExitStatus
	if err := ctrd.Restart(func() {
		exitStatus = <-statusC
	}); err != nil {
		t.Fatal(err)
	}

	if exitStatus.Error() == nil {
		t.Errorf(`first task.Wait() should have failed with "transport is closing"`)
	}

	waitCtx, waitCancel := context.WithTimeout(ctx, 2*time.Second)
	serving, err := client.IsServing(waitCtx)
	waitCancel()
	if !serving {
		t.Fatalf("containerd did not start within 2s: %v", err)
	}

	statusC, err = task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	if err := task.Kill(ctx, syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}

	<-statusC
}

func TestContainerAttach(t *testing.T) {
	t.Parallel()

	if runtime.GOOS == "windows" {
		// On windows, closing the write side of the pipe closes the read
		// side, sending an EOF to it and preventing reopening it.
		// Hence this test will always fails on windows
		t.Skip("invalid logic on windows")
	}

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}

	container, err := client.NewContainer(ctx, id, WithNewSpec(withImageConfig(image), withCat()), withNewSnapshot(id, image))
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	expected := "hello" + newLine

	direct, err := NewDirectIO(ctx, false)
	if err != nil {
		t.Error(err)
		return
	}
	defer direct.Delete()
	var (
		wg  sync.WaitGroup
		buf = bytes.NewBuffer(nil)
	)
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(buf, direct.Stdout)
	}()

	task, err := container.NewTask(ctx, direct.IOCreate)
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	status, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
	}

	if err := task.Start(ctx); err != nil {
		t.Error(err)
		return
	}

	if _, err := fmt.Fprint(direct.Stdin, expected); err != nil {
		t.Error(err)
	}

	// load the container and re-load the task
	if container, err = client.LoadContainer(ctx, id); err != nil {
		t.Error(err)
		return
	}

	if task, err = container.Task(ctx, direct.IOAttach); err != nil {
		t.Error(err)
		return
	}

	if _, err := fmt.Fprint(direct.Stdin, expected); err != nil {
		t.Error(err)
	}

	direct.Stdin.Close()

	if err := task.CloseIO(ctx, WithStdinCloser); err != nil {
		t.Error(err)
	}

	<-status

	wg.Wait()
	if _, err := task.Delete(ctx); err != nil {
		t.Error(err)
	}

	output := buf.String()

	// we wrote the same thing after attach
	expected = expected + expected
	if output != expected {
		t.Errorf("expected output %q but received %q", expected, output)
	}
}

func TestContainerUsername(t *testing.T) {
	t.Parallel()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}
	direct, err := NewDirectIO(ctx, false)
	if err != nil {
		t.Error(err)
		return
	}
	defer direct.Delete()
	var (
		wg  sync.WaitGroup
		buf = bytes.NewBuffer(nil)
	)
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(buf, direct.Stdout)
	}()

	// squid user in the alpine image has a uid of 31
	container, err := client.NewContainer(ctx, id,
		withNewSnapshot(id, image),
		WithNewSpec(withImageConfig(image), WithUsername("squid"), WithProcessArgs("id", "-u")),
	)
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	task, err := container.NewTask(ctx, direct.IOCreate)
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	if err := task.Start(ctx); err != nil {
		t.Error(err)
		return
	}
	<-statusC

	wg.Wait()

	output := strings.TrimSuffix(buf.String(), "\n")
	if output != "31" {
		t.Errorf("expected squid uid to be 31 but received %q", output)
	}
}

func TestContainerAttachProcess(t *testing.T) {
	t.Parallel()

	if runtime.GOOS == "windows" {
		// On windows, closing the write side of the pipe closes the read
		// side, sending an EOF to it and preventing reopening it.
		// Hence this test will always fails on windows
		t.Skip("invalid logic on windows")
	}

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}

	container, err := client.NewContainer(ctx, id, WithNewSpec(withImageConfig(image), withProcessArgs("sleep", "100")), withNewSnapshot(id, image))
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	expected := "hello" + newLine

	// creating IO early for easy resource cleanup
	direct, err := NewDirectIO(ctx, false)
	if err != nil {
		t.Error(err)
		return
	}
	defer direct.Delete()
	var (
		wg  sync.WaitGroup
		buf = bytes.NewBuffer(nil)
	)
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(buf, direct.Stdout)
	}()

	task, err := container.NewTask(ctx, empty())
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	status, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
	}

	if err := task.Start(ctx); err != nil {
		t.Error(err)
		return
	}

	spec, err := container.Spec(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	processSpec := spec.Process
	processSpec.Args = []string{"cat"}
	execID := t.Name() + "_exec"
	process, err := task.Exec(ctx, execID, processSpec, direct.IOCreate)
	if err != nil {
		t.Error(err)
		return
	}
	processStatusC, err := process.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	if err := process.Start(ctx); err != nil {
		t.Error(err)
		return
	}

	if _, err := fmt.Fprint(direct.Stdin, expected); err != nil {
		t.Error(err)
	}

	if process, err = task.LoadProcess(ctx, execID, direct.IOAttach); err != nil {
		t.Error(err)
		return
	}

	if _, err := fmt.Fprint(direct.Stdin, expected); err != nil {
		t.Error(err)
	}

	direct.Stdin.Close()

	if err := process.CloseIO(ctx, WithStdinCloser); err != nil {
		t.Error(err)
	}

	<-processStatusC

	wg.Wait()

	if err := task.Kill(ctx, syscall.SIGKILL); err != nil {
		t.Error(err)
	}

	output := buf.String()

	// we wrote the same thing after attach
	expected = expected + expected
	if output != expected {
		t.Errorf("expected output %q but received %q", expected, output)
	}
	<-status
}

func TestContainerUserID(t *testing.T) {
	t.Parallel()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}
	direct, err := NewDirectIO(ctx, false)
	if err != nil {
		t.Error(err)
		return
	}
	defer direct.Delete()
	var (
		wg  sync.WaitGroup
		buf = bytes.NewBuffer(nil)
	)
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(buf, direct.Stdout)
	}()

	// adm user in the alpine image has a uid of 3 and gid of 4.
	container, err := client.NewContainer(ctx, id,
		withNewSnapshot(id, image),
		WithNewSpec(withImageConfig(image), WithUserID(3), WithProcessArgs("sh", "-c", "echo $(id -u):$(id -g)")),
	)
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	task, err := container.NewTask(ctx, direct.IOCreate)
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	if err := task.Start(ctx); err != nil {
		t.Error(err)
		return
	}
	<-statusC

	wg.Wait()

	output := strings.TrimSuffix(buf.String(), "\n")
	if output != "3:4" {
		t.Errorf("expected uid:gid to be 3:4, but received %q", output)
	}
}

func TestContainerKillAll(t *testing.T) {
	t.Parallel()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}

	container, err := client.NewContainer(ctx, id,
		withNewSnapshot(id, image),
		WithNewSpec(withImageConfig(image),
			withProcessArgs("sh", "-c", "top"),
			WithHostNamespace(specs.PIDNamespace),
		),
	)
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	stdout := bytes.NewBuffer(nil)
	task, err := container.NewTask(ctx, NewIO(bytes.NewBuffer(nil), stdout, bytes.NewBuffer(nil)))
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	if err := task.Start(ctx); err != nil {
		t.Error(err)
		return
	}

	if err := task.Kill(ctx, syscall.SIGKILL, WithKillAll); err != nil {
		t.Error(err)
	}

	<-statusC
	if _, err := task.Delete(ctx); err != nil {
		t.Error(err)
		return
	}
}

func TestShimSigkilled(t *testing.T) {
	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	// redis unset its PDeathSignal making it a good candidate
	image, err = client.Pull(ctx, "docker.io/library/redis:alpine", WithPullUnpack)
	if err != nil {
		t.Fatal(err)
	}
	container, err := client.NewContainer(ctx, id, WithNewSpec(WithImageConfig(image)), withNewSnapshot(id, image))
	if err != nil {
		t.Fatal(err)
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	task, err := container.NewTask(ctx, empty())
	if err != nil {
		t.Fatal(err)
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
	}

	pid := task.Pid()
	if pid <= 0 {
		t.Fatalf("invalid task pid %d", pid)
	}

	if err := task.Start(ctx); err != nil {
		t.Fatal(err)
	}

	// SIGKILL the shim
	if err := exec.Command("pkill", "-KILL", "containerd-s").Run(); err != nil {
		t.Fatalf("failed to kill shim: %v", err)
	}

	<-statusC

	for i := 0; i < 10; i++ {
		if err := unix.Kill(int(pid), 0); err == unix.ESRCH {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err := unix.Kill(int(pid), 0); err != unix.ESRCH {
		t.Errorf("pid %d still exists", pid)
	}

}

func TestDaemonRestartWithRunningShim(t *testing.T) {
	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}
	container, err := client.NewContainer(ctx, id, WithNewSpec(WithImageConfig(image), WithProcessArgs("sleep", "100")), withNewSnapshot(id, image))
	if err != nil {
		t.Fatal(err)
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	task, err := container.NewTask(ctx, empty())
	if err != nil {
		t.Fatal(err)
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
	}

	pid := task.Pid()
	if pid <= 0 {
		t.Fatalf("invalid task pid %d", pid)
	}

	if err := task.Start(ctx); err != nil {
		t.Fatal(err)
	}

	var exitStatus ExitStatus
	if err := ctrd.Restart(func() {
		exitStatus = <-statusC
	}); err != nil {
		t.Fatal(err)
	}

	if exitStatus.Error() == nil {
		t.Errorf(`first task.Wait() should have failed with "transport is closing"`)
	}

	waitCtx, cancel := context.WithTimeout(ctx, 1*time.Second)
	c, err := ctrd.waitForStart(waitCtx)
	cancel()
	if err != nil {
		t.Fatal(err)
	}
	c.Close()

	statusC, err = task.Wait(ctx)
	if err != nil {
		t.Error(err)
	}

	if err := task.Kill(ctx, syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}

	<-statusC

	if err := unix.Kill(int(pid), 0); err != unix.ESRCH {
		t.Errorf("pid %d still exists", pid)
	}
}

func TestContainerRuntimeOptions(t *testing.T) {
	t.Parallel()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}

	container, err := client.NewContainer(
		ctx, id,
		WithNewSpec(withImageConfig(image), withExitStatus(7)),
		withNewSnapshot(id, image),
		WithRuntime("io.containerd.runtime.v1.linux", &runcopts.RuncOptions{Runtime: "no-runc"}),
	)
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	task, err := container.NewTask(ctx, empty())
	if err == nil {
		t.Errorf("task creation should have failed")
		task.Delete(ctx)
		return
	}
	if !strings.Contains(err.Error(), `"no-runc"`) {
		t.Errorf("task creation should have failed because of lack of executable. Instead failed with: %v", err.Error())
	}
}

func TestContainerKillInitPidHost(t *testing.T) {
	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}

	container, err := client.NewContainer(ctx, id,
		withNewSnapshot(id, image),
		WithNewSpec(withImageConfig(image),
			withProcessArgs("sh", "-c", "sleep 42; echo hi"),
			WithHostNamespace(specs.PIDNamespace),
		),
	)
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	stdout := bytes.NewBuffer(nil)
	task, err := container.NewTask(ctx, NewIO(bytes.NewBuffer(nil), stdout, bytes.NewBuffer(nil)))
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	if err := task.Start(ctx); err != nil {
		t.Error(err)
		return
	}

	if err := task.Kill(ctx, syscall.SIGKILL); err != nil {
		t.Error(err)
	}

	// Give the shim time to reap the init process and kill the orphans
	select {
	case <-statusC:
	case <-time.After(100 * time.Millisecond):
	}

	b, err := exec.Command("ps", "ax").CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}

	if strings.Contains(string(b), "sleep 42") {
		t.Fatalf("killing init didn't kill all its children:\n%v", string(b))
	}

	if _, err := task.Delete(ctx, WithProcessKill); err != nil {
		t.Error(err)
	}
}

func TestUserNamespaces(t *testing.T) {
	t.Parallel()
	t.Run("WritableRootFS", func(t *testing.T) { testUserNamespaces(t, false) })
	// see #1373 and runc#1572
	t.Run("ReadonlyRootFS", func(t *testing.T) { testUserNamespaces(t, true) })
}

func checkUserNS(t *testing.T) {
	cmd := exec.Command("true")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWUSER,
	}

	if err := cmd.Run(); err != nil {
		t.Skip("User namespaces are unavailable")
	}
}

func testUserNamespaces(t *testing.T, readonlyRootFS bool) {
	checkUserNS(t)

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = strings.Replace(t.Name(), "/", "-", -1)
	)
	defer cancel()

	image, err = client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}

	opts := []NewContainerOpts{WithNewSpec(withImageConfig(image),
		withExitStatus(7),
		WithUserNamespace(0, 1000, 10000),
	)}
	if readonlyRootFS {
		opts = append(opts, withRemappedSnapshotView(id, image, 1000, 1000))
	} else {
		opts = append(opts, withRemappedSnapshot(id, image, 1000, 1000))
	}

	container, err := client.NewContainer(ctx, id, opts...)
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	task, err := container.NewTask(ctx, Stdio, func(_ context.Context, client *Client, r *TaskInfo) error {
		r.Options = &runcopts.CreateOptions{
			IoUid: 1000,
			IoGid: 1000,
		}
		return nil
	})
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}

	if pid := task.Pid(); pid <= 0 {
		t.Errorf("invalid task pid %d", pid)
	}
	if err := task.Start(ctx); err != nil {
		t.Error(err)
		task.Delete(ctx)
		return
	}
	status := <-statusC
	code, _, err := status.Result()
	if err != nil {
		t.Error(err)
		return
	}
	if code != 7 {
		t.Errorf("expected status 7 from wait but received %d", code)
	}
	deleteStatus, err := task.Delete(ctx)
	if err != nil {
		t.Error(err)
		return
	}
	if ec := deleteStatus.ExitCode(); ec != 7 {
		t.Errorf("expected status 7 from delete but received %d", ec)
	}
}

func TestTaskResize(t *testing.T) {
	t.Parallel()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	var (
		image       Image
		ctx, cancel = testContext()
		id          = t.Name()
	)
	defer cancel()

	if runtime.GOOS != "windows" {
		image, err = client.GetImage(ctx, testImage)
		if err != nil {
			t.Error(err)
			return
		}
	}
	container, err := client.NewContainer(ctx, id, WithNewSpec(withImageConfig(image), withExitStatus(7)), withNewSnapshot(id, image))
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx, WithSnapshotCleanup)

	task, err := container.NewTask(ctx, empty())
	if err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err := task.Wait(ctx)
	if err != nil {
		t.Error(err)
		return
	}
	if err := task.Resize(ctx, 32, 32); err != nil {
		t.Error(err)
		return
	}
	task.Kill(ctx, syscall.SIGKILL)
	<-statusC
}

func TestContainerImage(t *testing.T) {
	t.Parallel()

	ctx, cancel := testContext()
	defer cancel()
	id := t.Name()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	image, err := client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}

	container, err := client.NewContainer(ctx, id, WithNewSpec(), WithImage(image))
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx)

	i, err := container.Image(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if i.Name() != image.Name() {
		t.Fatalf("expected container image name %s but received %s", image.Name(), i.Name())
	}
}

func TestContainerNoImage(t *testing.T) {
	t.Parallel()

	ctx, cancel := testContext()
	defer cancel()
	id := t.Name()

	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	container, err := client.NewContainer(ctx, id, WithNewSpec())
	if err != nil {
		t.Error(err)
		return
	}
	defer container.Delete(ctx)

	_, err = container.Image(ctx)
	if err == nil {
		t.Fatal("error should not be nil when container is created without an image")
	}
	if errors.Cause(err) != errdefs.ErrNotFound {
		t.Fatalf("expected error to be %s but received %s", errdefs.ErrNotFound, err)
	}
}
