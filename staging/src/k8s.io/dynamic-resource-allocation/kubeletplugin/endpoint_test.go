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

	g "github.com/onsi/gomega"
	"k8s.io/ktesting"
)

func TestEndpointLifecycle(t *testing.T) { testEndpointLifecycle(ktesting.Init(t)) }
func testEndpointLifecycle(tCtx ktesting.TContext) {
	tempDir := tCtx.TempDir()
	socketname := "test.sock"
	e := endpoint{dir: tempDir, file: socketname}
	listener, err := e.listen(tCtx)
	tCtx.ExpectNoError(err, "listen")
	tCtx.Assert(path.Join(tempDir, socketname)).To(g.BeAnExistingFile())
	tCtx.ExpectNoError(listener.Close(), "close")
	tCtx.Assert(path.Join(tempDir, socketname)).ToNot(g.BeAnExistingFile())
}

func TestEndpointListener(t *testing.T) { testEndpointListener(ktesting.Init(t)) }
func testEndpointListener(tCtx ktesting.TContext) {
	tempDir := tCtx.TempDir()
	socketname := "test.sock"
	listen := func(ctx2 context.Context, socketpath string) (net.Listener, error) {
		tCtx.Assert(socketpath).To(g.Equal(path.Join(tempDir, socketname)))
		return nil, nil
	}
	e := endpoint{dir: tempDir, file: socketname, listenFunc: listen}
	listener, err := e.listen(tCtx)
	tCtx.ExpectNoError(err, "listen")
	tCtx.Assert(path.Join(tempDir, socketname)).ToNot(g.BeAnExistingFile())
	tCtx.Assert(listener).To(g.BeNil())
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
func unremovableSocket(tCtx ktesting.TContext, dir, file string) {
	tCtx.Helper()
	tCtx.ExpectNoError(os.Mkdir(path.Join(dir, file), 0700))
	tCtx.ExpectNoError(os.WriteFile(path.Join(dir, file, "occupied"), nil, 0600))
}

func TestEndpointCloseReportsFailedSocketRemoval(t *testing.T) {
	testEndpointCloseReportsFailedSocketRemoval(ktesting.Init(t))
}
func testEndpointCloseReportsFailedSocketRemoval(tCtx ktesting.TContext) {
	tempDir := tCtx.TempDir()
	socketname := "test.sock"
	unremovableSocket(tCtx, tempDir, socketname)
	listener := &unixListener{
		Listener: closeErrorListener{},
		endpoint: endpoint{dir: tempDir, file: socketname},
	}

	err := listener.Close()

	tCtx.Require(err).To(g.MatchError(g.ContainSubstring("remove Unix domain socket")), "closing must report the socket that was left behind")
}

func TestEndpointCloseKeepsBothErrors(t *testing.T) {
	testEndpointCloseKeepsBothErrors(ktesting.Init(t))
}
func testEndpointCloseKeepsBothErrors(tCtx ktesting.TContext) {
	tempDir := tCtx.TempDir()
	socketname := "test.sock"
	unremovableSocket(tCtx, tempDir, socketname)
	closeErr := errors.New("close failed")
	listener := &unixListener{
		Listener: closeErrorListener{closeErr: closeErr},
		endpoint: endpoint{dir: tempDir, file: socketname},
	}

	err := listener.Close()

	tCtx.Require(err).To(g.MatchError(closeErr), "the listener's own error")
	tCtx.Assert(err.Error()).To(g.ContainSubstring("remove Unix domain socket"), "the removal error")
}
