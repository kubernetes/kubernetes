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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"syscall"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/httpstream/wsstream"
	"k8s.io/apimachinery/pkg/util/runtime"

	"k8s.io/klog/v2"
)

// portForwardErrResponse
// It will be sent to the client to instruct the client whether to close the Connection.
type portForwardErrResponse struct {
	// unexpected error was encountered and the connection needs to be closed.
	CloseConnection bool `json:"closeConnection,omitempty"`
	// error details
	// regardless of whether we encounter an expected error, we still need to set it up.
	Message string `json:"message,omitempty"`
}

// PortForwarder knows how to forward content from a data stream to/from a port
// in a pod.
type PortForwarder interface {
	// PortForwarder copies data between a data stream and a port in a pod.
	PortForward(ctx context.Context, name string, uid types.UID, port int32, stream io.ReadWriteCloser) error
}

// ServePortForward handles a port forwarding request.  A single request is
// kept alive as long as the client is still alive and the connection has not
// been timed out due to idleness. This function handles multiple forwarded
// connections; i.e., multiple `curl http://localhost:8888/` requests will be
// handled by a single invocation of ServePortForward.
func ServePortForward(w http.ResponseWriter, req *http.Request, portForwarder PortForwarder, podName string, uid types.UID, portForwardOptions *V4Options, idleTimeout time.Duration, streamCreationTimeout time.Duration, supportedProtocols []string) {
	var err error
	if wsstream.IsWebSocketRequest(req) {
		err = handleWebSocketStreams(req, w, portForwarder, podName, uid, portForwardOptions, supportedProtocols, idleTimeout, streamCreationTimeout)
	} else {
		err = handleHTTPStreams(req, w, portForwarder, podName, uid, supportedProtocols, idleTimeout, streamCreationTimeout)
	}

	if err != nil {
		runtime.HandleError(err)
		return
	}
}

// handleStreamPortForwardErr
// Whether it is httpStream or WebSocket, we use the same method to handle errors.
func handleStreamPortForwardErr(err error, pod string, port int32, uid types.UID, requestID string, conn any, errorStream io.WriteCloser) {
	// happy path, we have successfully completed forwarding task
	if err == nil {
		return
	}

	errResp := portForwardErrResponse{
		CloseConnection: false,
	}
	if errors.Is(err, syscall.EPIPE) || errors.Is(err, syscall.ECONNRESET) {
		// During the forwarding process, if these types of errors are encountered,
		// we should continue to provide port forwarding services.
		//
		// These two errors can occur in the following scenarios:
		// ECONNRESET: the target process reset connection between CRI and itself.
		// see: https://github.com/kubernetes/kubernetes/issues/111825 for detail
		//
		// EPIPE: the target process did not read the received data, causing the
		// buffer in the kernel to be full, resulting in the occurrence of Zero Window,
		// then closing the connection (FIN, RESET)
		// see: https://github.com/kubernetes/kubernetes/issues/74551 for detail
		klog.ErrorS(err, "forwarding port", "conn", conn, "request", requestID, "port", port)
	} else {
		errResp.CloseConnection = true
	}

	// send error messages to the client to let our users know what happened.
	msg := fmt.Errorf("error forwarding port %d to pod %s, uid %v: %w", port, pod, uid, err)
	errResp.Message = msg.Error()
	runtime.HandleError(msg)
	err = json.NewEncoder(errorStream).Encode(errResp)
	if err != nil {
		klog.ErrorS(err, "encode the resp", "conn", conn, "request", requestID, "port", port)
	}
}
