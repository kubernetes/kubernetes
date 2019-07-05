// +build linux

package libcontainer

import (
	"fmt"
	"os"

	"github.com/opencontainers/runc/libcontainer/system"
)

func newRestoredProcess(pid int, fds []string) (*restoredProcess, error) {
	var (
		err error
	)
	proc, err := os.FindProcess(pid)
	if err != nil {
		return nil, err
	}
	stat, err := system.Stat(pid)
	if err != nil {
		return nil, err
	}
	return &restoredProcess{
		proc:             proc,
		processStartTime: stat.StartTime,
		fds:              fds,
	}, nil
}

type restoredProcess struct {
	proc             *os.Process
	processStartTime uint64
	fds              []string
}

func (p *restoredProcess) start() error {
	return newGenericError(fmt.Errorf("restored process cannot be started"), SystemError)
}

func (p *restoredProcess) pid() int {
	return p.proc.Pid
}

func (p *restoredProcess) terminate() error {
	err := p.proc.Kill()
	if _, werr := p.wait(); err == nil {
		err = werr
	}
	return err
}

func (p *restoredProcess) wait() (*os.ProcessState, error) {
	// TODO: how do we wait on the actual process?
	// maybe use --exec-cmd in criu
	st, err := p.proc.Wait()
	if err != nil {
		return nil, err
	}
	return st, nil
}

func (p *restoredProcess) startTime() (uint64, error) {
	return p.processStartTime, nil
}

func (p *restoredProcess) signal(s os.Signal) error {
	return p.proc.Signal(s)
}

func (p *restoredProcess) externalDescriptors() []string {
	return p.fds
}

func (p *restoredProcess) setExternalDescriptors(newFds []string) {
	p.fds = newFds
}

func (p *restoredProcess) forwardChildLogs() {
}

// nonChildProcess represents a process where the calling process is not
// the parent process.  This process is created when a factory loads a container from
// a persisted state.
type nonChildProcess struct {
	processPid       int
	processStartTime uint64
	fds              []string
}

func (p *nonChildProcess) start() error {
	return newGenericError(fmt.Errorf("restored process cannot be started"), SystemError)
}

func (p *nonChildProcess) pid() int {
	return p.processPid
}

func (p *nonChildProcess) terminate() error {
	return newGenericError(fmt.Errorf("restored process cannot be terminated"), SystemError)
}

func (p *nonChildProcess) wait() (*os.ProcessState, error) {
	return nil, newGenericError(fmt.Errorf("restored process cannot be waited on"), SystemError)
}

func (p *nonChildProcess) startTime() (uint64, error) {
	return p.processStartTime, nil
}

func (p *nonChildProcess) signal(s os.Signal) error {
	proc, err := os.FindProcess(p.processPid)
	if err != nil {
		return err
	}
	return proc.Signal(s)
}

func (p *nonChildProcess) externalDescriptors() []string {
	return p.fds
}

func (p *nonChildProcess) setExternalDescriptors(newFds []string) {
	p.fds = newFds
}

func (p *nonChildProcess) forwardChildLogs() {
}
