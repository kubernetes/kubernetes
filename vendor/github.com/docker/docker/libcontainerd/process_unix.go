// +build linux solaris

package libcontainerd

import (
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	goruntime "runtime"
	"strings"

	containerd "github.com/containerd/containerd/api/grpc/types"
	"github.com/tonistiigi/fifo"
	"golang.org/x/net/context"
	"golang.org/x/sys/unix"
)

var fdNames = map[int]string{
	unix.Stdin:  "stdin",
	unix.Stdout: "stdout",
	unix.Stderr: "stderr",
}

// process keeps the state for both main container process and exec process.
type process struct {
	processCommon

	// Platform specific fields are below here.
	dir string
}

func (p *process) openFifos(ctx context.Context, terminal bool) (pipe *IOPipe, err error) {
	if err := os.MkdirAll(p.dir, 0700); err != nil {
		return nil, err
	}

	io := &IOPipe{}

	io.Stdin, err = fifo.OpenFifo(ctx, p.fifo(unix.Stdin), unix.O_WRONLY|unix.O_CREAT|unix.O_NONBLOCK, 0700)
	if err != nil {
		return nil, err
	}

	defer func() {
		if err != nil {
			io.Stdin.Close()
		}
	}()

	io.Stdout, err = fifo.OpenFifo(ctx, p.fifo(unix.Stdout), unix.O_RDONLY|unix.O_CREAT|unix.O_NONBLOCK, 0700)
	if err != nil {
		return nil, err
	}

	defer func() {
		if err != nil {
			io.Stdout.Close()
		}
	}()

	if goruntime.GOOS == "solaris" || !terminal {
		// For Solaris terminal handling is done exclusively by the runtime therefore we make no distinction
		// in the processing for terminal and !terminal cases.
		io.Stderr, err = fifo.OpenFifo(ctx, p.fifo(unix.Stderr), unix.O_RDONLY|unix.O_CREAT|unix.O_NONBLOCK, 0700)
		if err != nil {
			return nil, err
		}
		defer func() {
			if err != nil {
				io.Stderr.Close()
			}
		}()
	} else {
		io.Stderr = ioutil.NopCloser(emptyReader{})
	}

	return io, nil
}

func (p *process) sendCloseStdin() error {
	_, err := p.client.remote.apiClient.UpdateProcess(context.Background(), &containerd.UpdateProcessRequest{
		Id:         p.containerID,
		Pid:        p.friendlyName,
		CloseStdin: true,
	})
	if err != nil && (strings.Contains(err.Error(), "container not found") || strings.Contains(err.Error(), "process not found")) {
		return nil
	}
	return err
}

func (p *process) closeFifos(io *IOPipe) {
	io.Stdin.Close()
	io.Stdout.Close()
	io.Stderr.Close()
}

type emptyReader struct{}

func (r emptyReader) Read(b []byte) (int, error) {
	return 0, io.EOF
}

func (p *process) fifo(index int) string {
	return filepath.Join(p.dir, p.friendlyName+"-"+fdNames[index])
}
