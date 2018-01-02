// +build !windows

package shim

import (
	"context"

	"github.com/containerd/console"
	shimapi "github.com/containerd/containerd/linux/shim/v1"
	"github.com/pkg/errors"
)

type deletedState struct {
}

func (s *deletedState) Pause(ctx context.Context) error {
	return errors.Errorf("cannot pause a deleted process")
}

func (s *deletedState) Resume(ctx context.Context) error {
	return errors.Errorf("cannot resume a deleted process")
}

func (s *deletedState) Update(context context.Context, r *shimapi.UpdateTaskRequest) error {
	return errors.Errorf("cannot update a deleted process")
}

func (s *deletedState) Checkpoint(ctx context.Context, r *shimapi.CheckpointTaskRequest) error {
	return errors.Errorf("cannot checkpoint a deleted process")
}

func (s *deletedState) Resize(ws console.WinSize) error {
	return errors.Errorf("cannot resize a deleted process")
}

func (s *deletedState) Start(ctx context.Context) error {
	return errors.Errorf("cannot start a deleted process")
}

func (s *deletedState) Delete(ctx context.Context) error {
	return errors.Errorf("cannot delete a deleted process")
}

func (s *deletedState) Kill(ctx context.Context, sig uint32, all bool) error {
	return errors.Errorf("cannot kill a deleted process")
}

func (s *deletedState) SetExited(status int) {
	// no op
}
