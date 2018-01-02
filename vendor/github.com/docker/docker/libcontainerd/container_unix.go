// +build linux solaris

package libcontainerd

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"

	containerd "github.com/containerd/containerd/api/grpc/types"
	"github.com/docker/docker/pkg/ioutils"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"github.com/tonistiigi/fifo"
	"golang.org/x/net/context"
	"golang.org/x/sys/unix"
)

type container struct {
	containerCommon

	// Platform specific fields are below here.
	pauseMonitor
	oom         bool
	runtime     string
	runtimeArgs []string
}

type runtime struct {
	path string
	args []string
}

// WithRuntime sets the runtime to be used for the created container
func WithRuntime(path string, args []string) CreateOption {
	return runtime{path, args}
}

func (rt runtime) Apply(p interface{}) error {
	if pr, ok := p.(*container); ok {
		pr.runtime = rt.path
		pr.runtimeArgs = rt.args
	}
	return nil
}

func (ctr *container) clean() error {
	if os.Getenv("LIBCONTAINERD_NOCLEAN") == "1" {
		return nil
	}
	if _, err := os.Lstat(ctr.dir); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	if err := os.RemoveAll(ctr.dir); err != nil {
		return err
	}
	return nil
}

// cleanProcess removes the fifos used by an additional process.
// Caller needs to lock container ID before calling this method.
func (ctr *container) cleanProcess(id string) {
	if p, ok := ctr.processes[id]; ok {
		for _, i := range []int{unix.Stdin, unix.Stdout, unix.Stderr} {
			if err := os.Remove(p.fifo(i)); err != nil && !os.IsNotExist(err) {
				logrus.Warnf("libcontainerd: failed to remove %v for process %v: %v", p.fifo(i), id, err)
			}
		}
	}
	delete(ctr.processes, id)
}

func (ctr *container) spec() (*specs.Spec, error) {
	var spec specs.Spec
	dt, err := ioutil.ReadFile(filepath.Join(ctr.dir, configFilename))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(dt, &spec); err != nil {
		return nil, err
	}
	return &spec, nil
}

func (ctr *container) start(spec *specs.Spec, checkpoint, checkpointDir string, attachStdio StdioCallback) (err error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ready := make(chan struct{})

	fifoCtx, cancel := context.WithCancel(context.Background())
	defer func() {
		if err != nil {
			cancel()
		}
	}()

	iopipe, err := ctr.openFifos(fifoCtx, spec.Process.Terminal)
	if err != nil {
		return err
	}

	var stdinOnce sync.Once

	// we need to delay stdin closure after container start or else "stdin close"
	// event will be rejected by containerd.
	// stdin closure happens in attachStdio
	stdin := iopipe.Stdin
	iopipe.Stdin = ioutils.NewWriteCloserWrapper(stdin, func() error {
		var err error
		stdinOnce.Do(func() { // on error from attach we don't know if stdin was already closed
			err = stdin.Close()
			go func() {
				select {
				case <-ready:
				case <-ctx.Done():
				}
				select {
				case <-ready:
					if err := ctr.sendCloseStdin(); err != nil {
						logrus.Warnf("failed to close stdin: %+v", err)
					}
				default:
				}
			}()
		})
		return err
	})

	r := &containerd.CreateContainerRequest{
		Id:            ctr.containerID,
		BundlePath:    ctr.dir,
		Stdin:         ctr.fifo(unix.Stdin),
		Stdout:        ctr.fifo(unix.Stdout),
		Stderr:        ctr.fifo(unix.Stderr),
		Checkpoint:    checkpoint,
		CheckpointDir: checkpointDir,
		// check to see if we are running in ramdisk to disable pivot root
		NoPivotRoot: os.Getenv("DOCKER_RAMDISK") != "",
		Runtime:     ctr.runtime,
		RuntimeArgs: ctr.runtimeArgs,
	}
	ctr.client.appendContainer(ctr)

	if err := attachStdio(*iopipe); err != nil {
		ctr.closeFifos(iopipe)
		return err
	}

	resp, err := ctr.client.remote.apiClient.CreateContainer(context.Background(), r)
	if err != nil {
		ctr.closeFifos(iopipe)
		return err
	}
	ctr.systemPid = systemPid(resp.Container)
	close(ready)

	return ctr.client.backend.StateChanged(ctr.containerID, StateInfo{
		CommonStateInfo: CommonStateInfo{
			State: StateStart,
			Pid:   ctr.systemPid,
		}})

}

func (ctr *container) newProcess(friendlyName string) *process {
	return &process{
		dir: ctr.dir,
		processCommon: processCommon{
			containerID:  ctr.containerID,
			friendlyName: friendlyName,
			client:       ctr.client,
		},
	}
}

func (ctr *container) handleEvent(e *containerd.Event) error {
	ctr.client.lock(ctr.containerID)
	defer ctr.client.unlock(ctr.containerID)
	switch e.Type {
	case StateExit, StatePause, StateResume, StateOOM:
		st := StateInfo{
			CommonStateInfo: CommonStateInfo{
				State:    e.Type,
				ExitCode: e.Status,
			},
			OOMKilled: e.Type == StateExit && ctr.oom,
		}
		if e.Type == StateOOM {
			ctr.oom = true
		}
		if e.Type == StateExit && e.Pid != InitFriendlyName {
			st.ProcessID = e.Pid
			st.State = StateExitProcess
		}

		// Remove process from list if we have exited
		switch st.State {
		case StateExit:
			ctr.clean()
			ctr.client.deleteContainer(e.Id)
		case StateExitProcess:
			ctr.cleanProcess(st.ProcessID)
		}
		ctr.client.q.append(e.Id, func() {
			if err := ctr.client.backend.StateChanged(e.Id, st); err != nil {
				logrus.Errorf("libcontainerd: backend.StateChanged(): %v", err)
			}
			if e.Type == StatePause || e.Type == StateResume {
				ctr.pauseMonitor.handle(e.Type)
			}
			if e.Type == StateExit {
				if en := ctr.client.getExitNotifier(e.Id); en != nil {
					en.close()
				}
			}
		})

	default:
		logrus.Debugf("libcontainerd: event unhandled: %+v", e)
	}
	return nil
}

// discardFifos attempts to fully read the container fifos to unblock processes
// that may be blocked on the writer side.
func (ctr *container) discardFifos() {
	ctx, _ := context.WithTimeout(context.Background(), 3*time.Second)
	for _, i := range []int{unix.Stdout, unix.Stderr} {
		f, err := fifo.OpenFifo(ctx, ctr.fifo(i), unix.O_RDONLY|unix.O_NONBLOCK, 0)
		if err != nil {
			logrus.Warnf("error opening fifo %v for discarding: %+v", f, err)
			continue
		}
		go func() {
			io.Copy(ioutil.Discard, f)
		}()
	}
}
