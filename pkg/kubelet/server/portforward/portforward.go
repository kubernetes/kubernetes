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

package portforward

import (
	"io"
	"net/http"
	"time"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wsstream"
)

const (
	NonZeroExitCodeReason = unversioned.StatusReason("NonZeroExitCode")
	ExitCodeCauseType     = unversioned.CauseType("ExitCode")
)

// PortForwarder knows how to forward content from a data stream to/from a port
// in a pod.
type PortForwarder interface {
	// PortForwarder copies data between a data stream and a port in a pod.
	PortForward(name string, uid types.UID, port uint16, stream io.ReadWriteCloser) error
}

// context contains the connection and streams used when
// port forwarding from a container.
type context struct {
	conn io.Closer
}

// ServePortForward handles a port forwarding request.  A single request is
// kept alive as long as the client is still alive and the connection has not
// been timed out due to idleness. This function handles multiple forwarded
// connections; i.e., multiple `curl http://localhost:8888/` requests will be
// handled by a single invocation of ServePortForward.
func ServePortForward(w http.ResponseWriter, req *http.Request, portForwarder PortForwarder, podName string, uid types.UID, idleTimeout time.Duration, streamCreationTimeout time.Duration, supportedProtocols []string) {

	_, ok := createStreams(req, w, portForwarder, podName, uid, supportedProtocols, idleTimeout, streamCreationTimeout)
	if !ok {
		// error is handled by createStreams
		return
	}

}

func createStreams(req *http.Request, w http.ResponseWriter, portForwarder PortForwarder, podName string, uid types.UID, supportedStreamProtocols []string, idleTimeout, streamCreationTimeout time.Duration) (*context, bool) {
	var ctx *context
	var err error
	if wsstream.IsWebSocketRequest(req) {
		ctx, err = createWebSocketStreams(req, w, portForwarder, podName, uid, supportedStreamProtocols, idleTimeout, streamCreationTimeout)
	} else {
		ctx, err = createHttpStreamStreams(req, w, portForwarder, podName, uid, supportedStreamProtocols, idleTimeout, streamCreationTimeout)
	}

	if err != nil {
		runtime.HandleError(err)
		return nil, false
	}

	return ctx, true
}
