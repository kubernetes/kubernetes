// +build !windows

package containerd

import (
	"syscall"
	"testing"
)

func TestCheckpointRestore(t *testing.T) {
	if !supportsCriu {
		t.Skip("system does not have criu installed")
	}
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
	container, err := client.NewContainer(ctx, id, WithNewSpec(WithImageConfig(image), WithProcessArgs("sleep", "100")), WithNewSnapshot(id, image))
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

	checkpoint, err := task.Checkpoint(ctx, WithExit)
	if err != nil {
		t.Error(err)
		return
	}

	<-statusC

	if _, err := task.Delete(ctx); err != nil {
		t.Error(err)
		return
	}
	if task, err = container.NewTask(ctx, empty(), WithTaskCheckpoint(checkpoint)); err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err = task.Wait(ctx)
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
		return
	}
	<-statusC
}

func TestCheckpointRestoreNewContainer(t *testing.T) {
	if !supportsCriu {
		t.Skip("system does not have criu installed")
	}
	client, err := newClient(t, address)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	id := t.Name()
	ctx, cancel := testContext()
	defer cancel()

	image, err := client.GetImage(ctx, testImage)
	if err != nil {
		t.Error(err)
		return
	}
	container, err := client.NewContainer(ctx, id, WithNewSpec(WithImageConfig(image), WithProcessArgs("sleep", "100")), WithNewSnapshot(id, image))
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

	checkpoint, err := task.Checkpoint(ctx, WithExit)
	if err != nil {
		t.Error(err)
		return
	}

	<-statusC

	if _, err := task.Delete(ctx); err != nil {
		t.Error(err)
		return
	}
	if err := container.Delete(ctx, WithSnapshotCleanup); err != nil {
		t.Error(err)
		return
	}
	if container, err = client.NewContainer(ctx, id, WithCheckpoint(checkpoint, id)); err != nil {
		t.Error(err)
		return
	}
	if task, err = container.NewTask(ctx, empty(), WithTaskCheckpoint(checkpoint)); err != nil {
		t.Error(err)
		return
	}
	defer task.Delete(ctx)

	statusC, err = task.Wait(ctx)
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
		return
	}
	<-statusC
}

func TestCheckpointLeaveRunning(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	if !supportsCriu {
		t.Skip("system does not have criu installed")
	}
	client, err := New(address)
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
	container, err := client.NewContainer(ctx, id, WithNewSpec(WithImageConfig(image), WithProcessArgs("sleep", "100")), WithNewSnapshot(id, image))
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

	if _, err := task.Checkpoint(ctx); err != nil {
		t.Error(err)
		return
	}

	status, err := task.Status(ctx)
	if err != nil {
		t.Error(err)
		return
	}
	if status.Status != Running {
		t.Errorf("expected status %q but received %q", Running, status)
		return
	}

	if err := task.Kill(ctx, syscall.SIGKILL); err != nil {
		t.Error(err)
		return
	}

	<-statusC
}
