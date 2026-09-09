/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package remotecommand

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"k8s.io/streaming/pkg/runtime"
)

// ExitError is returned by [Executor.ExecInContainer] when the command
// terminates with a non-zero exit code.
//
// ExitCode returns the command's exit code, or -1 if no exit code is
// available, for example if the process was terminated by a signal.
//
// [*os/exec.ExitError] satisfies this interface.
type ExitError interface {
	error
	ExitCode() int
}

// Executor knows how to execute a command in a container in a pod.
type Executor interface {
	// ExecInContainer executes a command in a container in the pod, copying data
	// between in/out/err and the container's stdin/stdout/stderr.
	//
	// If the command terminates with a non-zero exit code,
	// ExecInContainer should return an [ExitError].
	ExecInContainer(ctx context.Context, name string, uid string, container string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan TerminalSize, timeout time.Duration) error
}

// legacyExitError matches [k8s.io/utils/exec.ExitError] for backward compatibility.
type legacyExitError interface {
	error
	Exited() bool
	ExitStatus() int
}

func exitCode(err error) (int, bool) {
	if exitErr, ok := errors.AsType[ExitError](err); ok {
		// ExitCode returns -1 if the process has not exited or was terminated
		// by a signal, so only accept non-negative exit codes.
		if code := exitErr.ExitCode(); code >= 0 {
			return code, true
		}
	}

	if exitErrLegacy, ok := errors.AsType[legacyExitError](err); ok && exitErrLegacy.Exited() {
		return exitErrLegacy.ExitStatus(), true
	}

	return 0, false
}

// ServeExec handles requests to execute a command in a container. After
// creating/receiving the required streams, it delegates the actual execution
// to the executor.
func ServeExec(w http.ResponseWriter, req *http.Request, executor Executor, podName string, uid string, container string, cmd []string, streamOpts *Options, idleTimeout, streamCreationTimeout time.Duration, supportedProtocols []string) {
	ctx, ok := createStreams(req, w, streamOpts, supportedProtocols, idleTimeout, streamCreationTimeout)
	if !ok {
		// error is handled by createStreams
		return
	}
	defer ctx.conn.Close()

	err := executor.ExecInContainer(req.Context(), podName, uid, container, cmd, ctx.stdinStream, ctx.stdoutStream, ctx.stderrStream, ctx.tty, ctx.resizeChan, 0)
	if err != nil {
		if rc, ok := exitCode(err); ok {
			_ = ctx.writeStatus(&streamStatusError{ErrStatus: streamStatus{
				Status: statusFailure,
				Reason: NonZeroExitCodeReason,
				Details: &streamStatusDetails{
					Causes: []streamStatusCause{
						{
							Type:    ExitCodeCauseType,
							Message: strconv.Itoa(rc),
						},
					},
				},
				Message: fmt.Sprintf("command terminated with non-zero exit code: %v", err),
			}})
		} else {
			err = fmt.Errorf("error executing command in container: %v", err)
			runtime.HandleError(err)
			_ = ctx.writeStatus(newInternalError(err))
		}
	} else {
		_ = ctx.writeStatus(&streamStatusError{ErrStatus: streamStatus{
			Status: statusSuccess,
		}})
	}
}
