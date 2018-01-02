package containerd

import (
	"context"
	"io"
	"os/exec"
	"sync"
	"syscall"

	"github.com/pkg/errors"
)

type daemon struct {
	sync.Mutex
	addr string
	cmd  *exec.Cmd
}

func (d *daemon) start(name, address string, args []string, stdout, stderr io.Writer) error {
	d.Lock()
	defer d.Unlock()
	if d.cmd != nil {
		return errors.New("daemon is already running")
	}
	args = append(args, []string{"--address", address}...)
	cmd := exec.Command(name, args...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Start(); err != nil {
		cmd.Wait()
		return errors.Wrap(err, "failed to start daemon")
	}
	d.addr = address
	d.cmd = cmd
	return nil
}

func (d *daemon) waitForStart(ctx context.Context) (*Client, error) {
	var (
		client  *Client
		serving bool
		err     error
	)

	client, err = New(address)
	if err != nil {
		return nil, err
	}
	serving, err = client.IsServing(ctx)
	if !serving {
		client.Close()
		if err == nil {
			err = errors.New("connection was successful but service is not available")
		}
		return nil, err
	}
	return client, err
}

func (d *daemon) Stop() error {
	d.Lock()
	defer d.Unlock()
	if d.cmd == nil {
		return errors.New("daemon is not running")
	}
	return d.cmd.Process.Signal(syscall.SIGTERM)
}

func (d *daemon) Kill() error {
	d.Lock()
	defer d.Unlock()
	if d.cmd == nil {
		return errors.New("daemon is not running")
	}
	return d.cmd.Process.Kill()
}

func (d *daemon) Wait() error {
	d.Lock()
	defer d.Unlock()
	if d.cmd == nil {
		return errors.New("daemon is not running")
	}
	err := d.cmd.Wait()
	d.cmd = nil
	return err
}

func (d *daemon) Restart(stopCb func()) error {
	d.Lock()
	defer d.Unlock()
	if d.cmd == nil {
		return errors.New("daemon is not running")
	}

	var err error
	if err = d.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		return errors.Wrap(err, "failed to signal daemon")
	}

	d.cmd.Wait()

	if stopCb != nil {
		stopCb()
	}

	cmd := exec.Command(d.cmd.Path, d.cmd.Args[1:]...)
	cmd.Stdout = d.cmd.Stdout
	cmd.Stderr = d.cmd.Stderr
	if err := cmd.Start(); err != nil {
		cmd.Wait()
		return errors.Wrap(err, "failed to start new daemon instance")
	}
	d.cmd = cmd

	return nil
}
