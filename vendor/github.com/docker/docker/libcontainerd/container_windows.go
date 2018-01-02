package libcontainerd

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"time"

	"github.com/Microsoft/hcsshim"
	"github.com/docker/docker/pkg/system"
	"github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/windows"
)

type container struct {
	containerCommon

	// Platform specific fields are below here. There are none presently on Windows.
	options []CreateOption

	// The ociSpec is required, as client.Create() needs a spec,
	// but can be called from the RestartManager context which does not
	// otherwise have access to the Spec
	ociSpec specs.Spec

	manualStopRequested bool
	hcsContainer        hcsshim.Container
}

func (ctr *container) newProcess(friendlyName string) *process {
	return &process{
		processCommon: processCommon{
			containerID:  ctr.containerID,
			friendlyName: friendlyName,
			client:       ctr.client,
		},
	}
}

// start starts a created container.
// Caller needs to lock container ID before calling this method.
func (ctr *container) start(attachStdio StdioCallback) error {
	var err error
	isServicing := false

	for _, option := range ctr.options {
		if s, ok := option.(*ServicingOption); ok && s.IsServicing {
			isServicing = true
		}
	}

	// Start the container.  If this is a servicing container, this call will block
	// until the container is done with the servicing execution.
	logrus.Debugln("libcontainerd: starting container ", ctr.containerID)
	if err = ctr.hcsContainer.Start(); err != nil {
		logrus.Errorf("libcontainerd: failed to start container: %s", err)
		if err := ctr.terminate(); err != nil {
			logrus.Errorf("libcontainerd: failed to cleanup after a failed Start. %s", err)
		} else {
			logrus.Debugln("libcontainerd: cleaned up after failed Start by calling Terminate")
		}
		return err
	}

	// Note we always tell HCS to
	// create stdout as it's required regardless of '-i' or '-t' options, so that
	// docker can always grab the output through logs. We also tell HCS to always
	// create stdin, even if it's not used - it will be closed shortly. Stderr
	// is only created if it we're not -t.
	createProcessParms := &hcsshim.ProcessConfig{
		EmulateConsole:   ctr.ociSpec.Process.Terminal,
		WorkingDirectory: ctr.ociSpec.Process.Cwd,
		CreateStdInPipe:  !isServicing,
		CreateStdOutPipe: !isServicing,
		CreateStdErrPipe: !ctr.ociSpec.Process.Terminal && !isServicing,
	}
	createProcessParms.ConsoleSize[0] = uint(ctr.ociSpec.Process.ConsoleSize.Height)
	createProcessParms.ConsoleSize[1] = uint(ctr.ociSpec.Process.ConsoleSize.Width)

	// Configure the environment for the process
	createProcessParms.Environment = setupEnvironmentVariables(ctr.ociSpec.Process.Env)
	if ctr.ociSpec.Platform.OS == "windows" {
		createProcessParms.CommandLine = strings.Join(ctr.ociSpec.Process.Args, " ")
	} else {
		createProcessParms.CommandArgs = ctr.ociSpec.Process.Args
	}
	createProcessParms.User = ctr.ociSpec.Process.User.Username

	// LCOW requires the raw OCI spec passed through HCS and onwards to GCS for the utility VM.
	if system.LCOWSupported() && ctr.ociSpec.Platform.OS == "linux" {
		ociBuf, err := json.Marshal(ctr.ociSpec)
		if err != nil {
			return err
		}
		ociRaw := json.RawMessage(ociBuf)
		createProcessParms.OCISpecification = &ociRaw
	}

	// Start the command running in the container.
	newProcess, err := ctr.hcsContainer.CreateProcess(createProcessParms)
	if err != nil {
		logrus.Errorf("libcontainerd: CreateProcess() failed %s", err)
		if err := ctr.terminate(); err != nil {
			logrus.Errorf("libcontainerd: failed to cleanup after a failed CreateProcess. %s", err)
		} else {
			logrus.Debugln("libcontainerd: cleaned up after failed CreateProcess by calling Terminate")
		}
		return err
	}

	pid := newProcess.Pid()

	// Save the hcs Process and PID
	ctr.process.friendlyName = InitFriendlyName
	ctr.process.hcsProcess = newProcess

	// If this is a servicing container, wait on the process synchronously here and
	// if it succeeds, wait for it cleanly shutdown and merge into the parent container.
	if isServicing {
		exitCode := ctr.waitProcessExitCode(&ctr.process)

		if exitCode != 0 {
			if err := ctr.terminate(); err != nil {
				logrus.Warnf("libcontainerd: terminating servicing container %s failed: %s", ctr.containerID, err)
			}
			return fmt.Errorf("libcontainerd: servicing container %s returned non-zero exit code %d", ctr.containerID, exitCode)
		}

		return ctr.hcsContainer.WaitTimeout(time.Minute * 5)
	}

	var stdout, stderr io.ReadCloser
	var stdin io.WriteCloser
	stdin, stdout, stderr, err = newProcess.Stdio()
	if err != nil {
		logrus.Errorf("libcontainerd: failed to get stdio pipes: %s", err)
		if err := ctr.terminate(); err != nil {
			logrus.Errorf("libcontainerd: failed to cleanup after a failed Stdio. %s", err)
		}
		return err
	}

	iopipe := &IOPipe{Terminal: ctr.ociSpec.Process.Terminal}

	iopipe.Stdin = createStdInCloser(stdin, newProcess)

	// Convert io.ReadClosers to io.Readers
	if stdout != nil {
		iopipe.Stdout = ioutil.NopCloser(&autoClosingReader{ReadCloser: stdout})
	}
	if stderr != nil {
		iopipe.Stderr = ioutil.NopCloser(&autoClosingReader{ReadCloser: stderr})
	}

	// Save the PID
	logrus.Debugf("libcontainerd: process started - PID %d", pid)
	ctr.systemPid = uint32(pid)

	// Spin up a go routine waiting for exit to handle cleanup
	go ctr.waitExit(&ctr.process, true)

	ctr.client.appendContainer(ctr)

	if err := attachStdio(*iopipe); err != nil {
		// OK to return the error here, as waitExit will handle tear-down in HCS
		return err
	}

	// Tell the docker engine that the container has started.
	si := StateInfo{
		CommonStateInfo: CommonStateInfo{
			State: StateStart,
			Pid:   ctr.systemPid, // Not sure this is needed? Double-check monitor.go in daemon BUGBUG @jhowardmsft
		}}
	logrus.Debugf("libcontainerd: start() completed OK, %+v", si)
	return ctr.client.backend.StateChanged(ctr.containerID, si)

}

// waitProcessExitCode will wait for the given process to exit and return its error code.
func (ctr *container) waitProcessExitCode(process *process) int {
	// Block indefinitely for the process to exit.
	err := process.hcsProcess.Wait()
	if err != nil {
		if herr, ok := err.(*hcsshim.ProcessError); ok && herr.Err != windows.ERROR_BROKEN_PIPE {
			logrus.Warnf("libcontainerd: Wait() failed (container may have been killed): %s", err)
		}
		// Fall through here, do not return. This ensures we attempt to continue the
		// shutdown in HCS and tell the docker engine that the process/container
		// has exited to avoid a container being dropped on the floor.
	}

	exitCode, err := process.hcsProcess.ExitCode()
	if err != nil {
		if herr, ok := err.(*hcsshim.ProcessError); ok && herr.Err != windows.ERROR_BROKEN_PIPE {
			logrus.Warnf("libcontainerd: unable to get exit code from container %s", ctr.containerID)
		}
		// Since we got an error retrieving the exit code, make sure that the code we return
		// doesn't incorrectly indicate success.
		exitCode = -1

		// Fall through here, do not return. This ensures we attempt to continue the
		// shutdown in HCS and tell the docker engine that the process/container
		// has exited to avoid a container being dropped on the floor.
	}

	return exitCode
}

// waitExit runs as a goroutine waiting for the process to exit. It's
// equivalent to (in the linux containerd world) where events come in for
// state change notifications from containerd.
func (ctr *container) waitExit(process *process, isFirstProcessToStart bool) error {
	logrus.Debugln("libcontainerd: waitExit() on pid", process.systemPid)

	exitCode := ctr.waitProcessExitCode(process)
	// Lock the container while removing the process/container from the list
	ctr.client.lock(ctr.containerID)

	if !isFirstProcessToStart {
		ctr.cleanProcess(process.friendlyName)
	} else {
		ctr.client.deleteContainer(ctr.containerID)
	}

	// Unlock here so other threads are unblocked
	ctr.client.unlock(ctr.containerID)

	// Assume the container has exited
	si := StateInfo{
		CommonStateInfo: CommonStateInfo{
			State:     StateExit,
			ExitCode:  uint32(exitCode),
			Pid:       process.systemPid,
			ProcessID: process.friendlyName,
		},
		UpdatePending: false,
	}

	// But it could have been an exec'd process which exited
	if !isFirstProcessToStart {
		si.State = StateExitProcess
	} else {
		// Pending updates is only applicable for WCOW
		if ctr.ociSpec.Platform.OS == "windows" {
			updatePending, err := ctr.hcsContainer.HasPendingUpdates()
			if err != nil {
				logrus.Warnf("libcontainerd: HasPendingUpdates() failed (container may have been killed): %s", err)
			} else {
				si.UpdatePending = updatePending
			}
		}

		logrus.Debugf("libcontainerd: shutting down container %s", ctr.containerID)
		if err := ctr.shutdown(); err != nil {
			logrus.Debugf("libcontainerd: failed to shutdown container %s", ctr.containerID)
		} else {
			logrus.Debugf("libcontainerd: completed shutting down container %s", ctr.containerID)
		}
		if err := ctr.hcsContainer.Close(); err != nil {
			logrus.Error(err)
		}
	}

	if err := process.hcsProcess.Close(); err != nil {
		logrus.Errorf("libcontainerd: hcsProcess.Close(): %v", err)
	}

	// Call into the backend to notify it of the state change.
	logrus.Debugf("libcontainerd: waitExit() calling backend.StateChanged %+v", si)
	if err := ctr.client.backend.StateChanged(ctr.containerID, si); err != nil {
		logrus.Error(err)
	}

	logrus.Debugf("libcontainerd: waitExit() completed OK, %+v", si)

	return nil
}

// cleanProcess removes process from the map.
// Caller needs to lock container ID before calling this method.
func (ctr *container) cleanProcess(id string) {
	delete(ctr.processes, id)
}

// shutdown shuts down the container in HCS
// Caller needs to lock container ID before calling this method.
func (ctr *container) shutdown() error {
	const shutdownTimeout = time.Minute * 5
	err := ctr.hcsContainer.Shutdown()
	if hcsshim.IsPending(err) {
		// Explicit timeout to avoid a (remote) possibility that shutdown hangs indefinitely.
		err = ctr.hcsContainer.WaitTimeout(shutdownTimeout)
	} else if hcsshim.IsAlreadyStopped(err) {
		err = nil
	}

	if err != nil {
		logrus.Debugf("libcontainerd: error shutting down container %s %v calling terminate", ctr.containerID, err)
		if err := ctr.terminate(); err != nil {
			return err
		}
		return err
	}

	return nil
}

// terminate terminates the container in HCS
// Caller needs to lock container ID before calling this method.
func (ctr *container) terminate() error {
	const terminateTimeout = time.Minute * 5
	err := ctr.hcsContainer.Terminate()

	if hcsshim.IsPending(err) {
		err = ctr.hcsContainer.WaitTimeout(terminateTimeout)
	} else if hcsshim.IsAlreadyStopped(err) {
		err = nil
	}

	if err != nil {
		logrus.Debugf("libcontainerd: error terminating container %s %v", ctr.containerID, err)
		return err
	}

	return nil
}
