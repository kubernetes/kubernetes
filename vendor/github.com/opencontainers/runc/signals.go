// +build linux

package main

import (
	"os"
	"os/signal"
	"syscall"

	"github.com/Sirupsen/logrus"
	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/utils"
)

const signalBufferSize = 2048

// newSignalHandler returns a signal handler for processing SIGCHLD and SIGWINCH signals
// while still forwarding all other signals to the process.
func newSignalHandler(tty *tty) *signalHandler {
	// ensure that we have a large buffer size so that we do not miss any signals
	// incase we are not processing them fast enough.
	s := make(chan os.Signal, signalBufferSize)
	// handle all signals for the process.
	signal.Notify(s)
	return &signalHandler{
		tty:     tty,
		signals: s,
	}
}

// exit models a process exit status with the pid and
// exit status.
type exit struct {
	pid    int
	status int
}

type signalHandler struct {
	signals chan os.Signal
	tty     *tty
}

// forward handles the main signal event loop forwarding, resizing, or reaping depending
// on the signal received.
func (h *signalHandler) forward(process *libcontainer.Process) (int, error) {
	// make sure we know the pid of our main process so that we can return
	// after it dies.
	pid1, err := process.Pid()
	if err != nil {
		return -1, err
	}
	// perform the initial tty resize.
	h.tty.resize()
	for s := range h.signals {
		switch s {
		case syscall.SIGWINCH:
			h.tty.resize()
		case syscall.SIGCHLD:
			exits, err := h.reap()
			if err != nil {
				logrus.Error(err)
			}
			for _, e := range exits {
				logrus.WithFields(logrus.Fields{
					"pid":    e.pid,
					"status": e.status,
				}).Debug("process exited")
				if e.pid == pid1 {
					// call Wait() on the process even though we already have the exit
					// status because we must ensure that any of the go specific process
					// fun such as flushing pipes are complete before we return.
					process.Wait()
					return e.status, nil
				}
			}
		default:
			logrus.Debugf("sending signal to process %s", s)
			if err := syscall.Kill(pid1, s.(syscall.Signal)); err != nil {
				logrus.Error(err)
			}
		}
	}
	return -1, nil
}

// reap runs wait4 in a loop until we have finished processing any existing exits
// then returns all exits to the main event loop for further processing.
func (h *signalHandler) reap() (exits []exit, err error) {
	var (
		ws  syscall.WaitStatus
		rus syscall.Rusage
	)
	for {
		pid, err := syscall.Wait4(-1, &ws, syscall.WNOHANG, &rus)
		if err != nil {
			if err == syscall.ECHILD {
				return exits, nil
			}
			return nil, err
		}
		if pid <= 0 {
			return exits, nil
		}
		exits = append(exits, exit{
			pid:    pid,
			status: utils.ExitStatus(ws),
		})
	}
}

func (h *signalHandler) Close() error {
	return h.tty.Close()
}
