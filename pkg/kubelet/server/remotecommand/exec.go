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
	"fmt"
	"io"
	"net/http"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/term"
)

const (
	NonZeroExitCodeReason = metav1.StatusReason("NonZeroExitCode")
	ExitCodeCauseType     = metav1.CauseType("ExitCode")
)

// Executor knows how to execute a command in a container in a pod.
type Executor interface {
	// ExecInContainer executes a command in a container in the pod, copying data
	// between in/out/err and the container's stdin/stdout/stderr.
	ExecInContainer(name string, uid types.UID, container string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan term.Size, timeout time.Duration) error
}

// ServeExec handles requests to execute a command in a container. After
// creating/receiving the required streams, it delegates the actual execution
// to the executor.
func ServeExec(w http.ResponseWriter, req *http.Request, executor Executor, podName string, uid types.UID, container string, cmd []string, streamOpts *Options, idleTimeout, streamCreationTimeout time.Duration, supportedProtocols []string) {
	ctx, ok := createStreams(req, w, streamOpts, supportedProtocols, idleTimeout, streamCreationTimeout)
	if !ok {
		// error is handled by createStreams
		return
	}
	defer ctx.conn.Close()

	err := executor.ExecInContainer(podName, uid, container, cmd, ctx.stdinStream, ctx.stdoutStream, ctx.stderrStream, ctx.tty, ctx.resizeChan, 0)
	if err != nil {
		if exitErr, ok := err.(utilexec.ExitError); ok && exitErr.Exited() {
			rc := exitErr.ExitStatus()
			ctx.writeStatus(&apierrors.StatusError{ErrStatus: metav1.Status{
				Status: metav1.StatusFailure,
				Reason: NonZeroExitCodeReason,
				Details: &metav1.StatusDetails{
					Causes: []metav1.StatusCause{
						{
							Type:    ExitCodeCauseType,
							Message: fmt.Sprintf("%d", rc),
						},
					},
				},
				Message: fmt.Sprintf("command terminated with non-zero exit code: %v", exitErr),
			}})
		} else {
			err = fmt.Errorf("error executing command in container: %v", err)
			runtime.HandleError(err)
			ctx.writeStatus(apierrors.NewInternalError(err))
		}
	} else {
		ctx.writeStatus(&apierrors.StatusError{ErrStatus: metav1.Status{
			Status: metav1.StatusSuccess,
		}})
	}
}
