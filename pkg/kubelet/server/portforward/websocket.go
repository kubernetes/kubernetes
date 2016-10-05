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
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/httplog"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wsstream"
)

const (
	dataChannel = iota
	errorChannel

	v4BinaryWebsocketProtocol = "v4." + wsstream.ChannelWebSocketProtocol
	v4Base64WebsocketProtocol = "v4." + wsstream.Base64ChannelWebSocketProtocol
)

// options contains details about which streams are required for
// port forwarding.
type v4Options struct {
	port uint16
}

// newOptions creates a new options from the Request.
func newV4Options(req *http.Request) (*v4Options, error) {
	portString := req.FormValue(api.PortHeader)
	if len(portString) == 0 {
		return nil, fmt.Errorf("%q is required", api.PortHeader)
	}
	port, err := strconv.ParseUint(portString, 10, 16)
	if err != nil {
		return nil, fmt.Errorf("unable to parse %q as a port: %v", portString, err)
	}
	if port < 1 {
		return nil, fmt.Errorf("port %q must be > 0", portString)
	}

	return &v4Options{
		port: uint16(port),
	}, nil
}

func handleWebSocketStreams(req *http.Request, w http.ResponseWriter, portForwarder PortForwarder, podName string, uid types.UID, supportedPortForwardProtocols []string, idleTimeout, streamCreationTimeout time.Duration) error {
	opts, err := newV4Options(req)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, err.Error())
		return err
	}

	channels := []wsstream.ChannelType{wsstream.ReadWriteChannel, wsstream.WriteChannel}
	conn := wsstream.NewConn(map[string]wsstream.ChannelProtocolConfig{
		"": {
			Binary:   true,
			Channels: channels,
		},
		v4BinaryWebsocketProtocol: {
			Binary:   true,
			Channels: channels,
		},
		v4Base64WebsocketProtocol: {
			Binary:   false,
			Channels: channels,
		},
	})
	conn.SetIdleTimeout(idleTimeout)
	_, streams, err := conn.Open(httplog.Unlogged(w), req)
	if err != nil {
		err = fmt.Errorf("Unable to upgrade websocket connection: %v", err)
		return err
	}
	defer conn.Close()

	streams[dataChannel].Write([]byte{})
	streams[errorChannel].Write([]byte{})

	h := &websocketStreamHandler{
		conn:      conn,
		port:      opts.port,
		streams:   streams,
		pod:       podName,
		uid:       uid,
		forwarder: portForwarder,
	}

	h.run()

	return nil
}

// websocketStreamHandler is capable of processing a single port forward
// request over a websocket connection
type websocketStreamHandler struct {
	conn      *wsstream.Conn
	port      uint16
	streams   []io.ReadWriteCloser
	pod       string
	uid       types.UID
	forwarder PortForwarder
}

// run invokes the websocketStreamHandler's forwarder.PortForward
// function for the given stream pair.
func (h *websocketStreamHandler) run() {
	defer h.streams[dataChannel].Close()
	defer h.streams[errorChannel].Close()
	defer h.conn.Close()

	glog.V(5).Infof("(conn=%p) invoking forwarder.PortForward for port %d", h.conn, h.port)
	err := h.forwarder.PortForward(h.pod, h.uid, h.port, h.streams[dataChannel])
	glog.V(5).Infof("(conn=%p) done invoking forwarder.PortForward for port %d", h.conn, h.port)

	if err != nil {
		msg := fmt.Errorf("error forwarding port %d to pod %s, uid %v: %v", h.port, h.pod, h.uid, err)
		runtime.HandleError(msg)
		fmt.Fprint(h.streams[errorChannel], msg.Error())
	}
}
