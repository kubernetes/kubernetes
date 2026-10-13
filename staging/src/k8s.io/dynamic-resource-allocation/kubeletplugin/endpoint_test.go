/*
Copyright 2025 The Kubernetes Authors.

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

package kubeletplugin

import (
	"context"
	"errors"
	"net"
	"os"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2/ktesting"
)

func TestEndpointLifecycle(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tempDir := t.TempDir()
	socketname := "test.sock"
	e := endpoint{dir: tempDir, file: socketname}
	listener, err := e.listen(ctx)
	require.NoError(t, err, "listen")
	assert.FileExists(t, path.Join(tempDir, socketname))
	require.NoError(t, listener.Close(), "close")
	assert.NoFileExists(t, path.Join(tempDir, socketname))
}

func TestEndpointListener(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tempDir := t.TempDir()
	socketname := "test.sock"
	listen := func(ctx2 context.Context, socketpath string) (net.Listener, error) {
		assert.Equal(t, path.Join(tempDir, socketname), socketpath)
		return nil, nil
	}
	e := endpoint{dir: tempDir, file: socketname, listenFunc: listen}
	listener, err := e.listen(ctx)
	require.NoError(t, err, "listen")
	assert.NoFileExists(t, path.Join(tempDir, socketname))
	assert.Nil(t, listener)
}

// closeErrorListener is a net.Listener whose Close returns a fixed error. Only
// Close is called here, so the embedded Listener is left nil.
type closeErrorListener struct {
	net.Listener
	closeErr error
}

func (l closeErrorListener) Close() error { return l.closeErr }

// unremovableSocket puts something at the socket path that os.Remove refuses to
// delete. A directory that is not empty fails without depending on file
// permissions or on which user runs the test.
func unremovableSocket(t *testing.T, dir, file string) {
	t.Helper()
	require.NoError(t, os.Mkdir(path.Join(dir, file), 0700))
	require.NoError(t, os.WriteFile(path.Join(dir, file, "occupied"), nil, 0600))
}

func TestEndpointCloseReportsFailedSocketRemoval(t *testing.T) {
	tempDir := t.TempDir()
	socketname := "test.sock"
	unremovableSocket(t, tempDir, socketname)
	listener := &unixListener{
		Listener: closeErrorListener{},
		endpoint: endpoint{dir: tempDir, file: socketname},
	}

	err := listener.Close()

	require.Error(t, err, "closing must report the socket that was left behind")
	assert.Contains(t, err.Error(), "remove Unix domain socket")
}

func TestEndpointCloseKeepsBothErrors(t *testing.T) {
	tempDir := t.TempDir()
	socketname := "test.sock"
	unremovableSocket(t, tempDir, socketname)
	closeErr := errors.New("close failed")
	listener := &unixListener{
		Listener: closeErrorListener{closeErr: closeErr},
		endpoint: endpoint{dir: tempDir, file: socketname},
	}

	err := listener.Close()

	require.Error(t, err)
	require.ErrorIs(t, err, closeErr, "the listener's own error")
	assert.Contains(t, err.Error(), "remove Unix domain socket", "the removal error")
}
