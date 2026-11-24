/*
Copyright The Kubernetes Authors.

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

package podexec

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/onsi/gomega"

	"k8s.io/client-go/util/exec"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

// ExecCommandInContainerWithRetry executes a command in the specified container,
// polling until PollShortTimeout on errors unless the command itself exited with
// a non-zero status code (exec.CodeExitError).
func ExecCommandInContainerWithRetry(ctx context.Context, f *framework.Framework, podName, containerName string, cmd ...string) (string, string, error) {
	return ExecWithRetry(ctx, framework.PollShortTimeout, framework.Poll, func() (string, string, error) {
		return e2epod.ExecCommandInContainerWithFullOutput(f, podName, containerName, cmd...)
	})
}

// ExecWithRetry invokes execFn, polling until timeout at pollInterval on errors
// unless execFn returns an exec.CodeExitError (indicating the command ran and
// exited non-zero), which fails immediately without retrying.
func ExecWithRetry(ctx context.Context, timeout, pollInterval time.Duration, execFn func() (string, string, error)) (string, string, error) {
	var stdout, stderr string
	err := framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (error, error) {
		var execErr error
		stdout, stderr, execErr = execFn()
		if execErr != nil {
			if _, ok := errors.AsType[exec.CodeExitError](execErr); ok {
				return nil, fmt.Errorf("command failed (stderr: %q): %w", stderr, execErr)
			}
			return fmt.Errorf("exec failed (stderr: %q): %w", stderr, execErr), nil
		}
		return nil, nil
	})).WithTimeout(timeout).WithPolling(pollInterval).Should(gomega.Succeed())
	if err != nil {
		return stdout, stderr, err
	}
	return stdout, stderr, nil
}
