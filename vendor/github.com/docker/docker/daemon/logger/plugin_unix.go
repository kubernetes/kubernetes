// +build linux solaris freebsd

package logger

import (
	"context"
	"io"

	"github.com/pkg/errors"
	"github.com/tonistiigi/fifo"
	"golang.org/x/sys/unix"
)

func openPluginStream(a *pluginAdapter) (io.WriteCloser, error) {
	f, err := fifo.OpenFifo(context.Background(), a.fifoPath, unix.O_WRONLY|unix.O_CREAT|unix.O_NONBLOCK, 0700)
	if err != nil {
		return nil, errors.Wrapf(err, "error creating i/o pipe for log plugin: %s", a.Name())
	}
	return f, nil
}
