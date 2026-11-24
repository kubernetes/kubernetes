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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/client-go/util/exec"
)

func TestExecWithRetry(t *testing.T) {
	t.Run("succeeds on first attempt", func(t *testing.T) {
		attempts := 0
		stdout, stderr, err := ExecWithRetry(context.Background(), 5*time.Second, 10*time.Millisecond, func() (string, string, error) {
			attempts++
			return "tmpfs 64 0 64 0% /cache", "", nil
		})
		require.NoError(t, err)
		assert.Equal(t, "tmpfs 64 0 64 0% /cache", stdout)
		assert.Empty(t, stderr)
		assert.Equal(t, 1, attempts)
	})

	t.Run("retries error until recovery", func(t *testing.T) {
		attempts := 0
		stdout, _, err := ExecWithRetry(context.Background(), 5*time.Second, 10*time.Millisecond, func() (string, string, error) {
			attempts++
			if attempts < 3 {
				return "", "", errors.New("error reading from error stream: read: connection reset by peer")
			}
			return "tmpfs 128 0 128 0% /cache", "", nil
		})
		require.NoError(t, err)
		assert.Equal(t, "tmpfs 128 0 128 0% /cache", stdout)
		assert.Equal(t, 3, attempts)
	})

	t.Run("stops immediately on non-zero command exit code", func(t *testing.T) {
		attempts := 0
		_, _, err := ExecWithRetry(context.Background(), 5*time.Second, 10*time.Millisecond, func() (string, string, error) {
			attempts++
			return "", "dd: /cache/largefile: No space left on device", exec.CodeExitError{Err: errors.New("command terminated with exit code 1"), Code: 1}
		})
		require.Error(t, err)
		assert.Equal(t, 1, attempts)
	})

	t.Run("fails after timeout when errors persist", func(t *testing.T) {
		_, _, err := ExecWithRetry(context.Background(), 50*time.Millisecond, 10*time.Millisecond, func() (string, string, error) {
			return "", "", errors.New("error reading from error stream: read: connection reset by peer")
		})
		require.Error(t, err)
	})
}
