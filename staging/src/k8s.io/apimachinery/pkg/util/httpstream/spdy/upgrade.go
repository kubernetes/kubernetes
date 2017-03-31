/*
Copyright 2015 The Kubernetes Authors.

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

package spdy

import (
	"bufio"
	"fmt"
	"net"
	"net/http"
	"strings"

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/runtime"
)

const HeaderSpdy31 = "SPDY/3.1"

// responseUpgrader knows how to upgrade HTTP responses. It
// implements the httpstream.ResponseUpgrader interface.
type responseUpgrader struct {
}

// connWrapper is used to wrap a hijacked connection its bufio.Reader. All
// calls will be handled directly by the underlying net.Conn with the exception
// of Read calls, which will read from the bufio.Reader. This ensures that data
// already inside the used bufio.Reader instance is also read.
type connWrapper struct {
	net.Conn
	bufReader *bufio.Reader
}

func (w *connWrapper) Read(b []byte) (n int, err error) {
	return w.bufReader.Read(b)
}

// NewResponseUpgrader returns a new httpstream.ResponseUpgrader that is
// capable of upgrading HTTP responses using SPDY/3.1 via the
// spdystream package.
func NewResponseUpgrader() httpstream.ResponseUpgrader {
	return responseUpgrader{}
}

// UpgradeResponse upgrades an HTTP response to one that supports multiplexed
// streams. newStreamHandler will be called synchronously whenever the
// other end of the upgraded connection creates a new stream.
func (u responseUpgrader) UpgradeResponse(w http.ResponseWriter, req *http.Request, newStreamHandler httpstream.NewStreamHandler) httpstream.Connection {
	connectionHeader := strings.ToLower(req.Header.Get(httpstream.HeaderConnection))
	upgradeHeader := strings.ToLower(req.Header.Get(httpstream.HeaderUpgrade))
	if !strings.Contains(connectionHeader, strings.ToLower(httpstream.HeaderUpgrade)) || !strings.Contains(upgradeHeader, strings.ToLower(HeaderSpdy31)) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "unable to upgrade: missing upgrade headers in request: %#v", req.Header)
		return nil
	}

	hijacker, ok := w.(http.Hijacker)
	if !ok {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, "unable to upgrade: unable to hijack response")
		return nil
	}

	w.Header().Add(httpstream.HeaderConnection, httpstream.HeaderUpgrade)
	w.Header().Add(httpstream.HeaderUpgrade, HeaderSpdy31)
	w.WriteHeader(http.StatusSwitchingProtocols)

	conn, bufrw, err := hijacker.Hijack()
	if err != nil {
		runtime.HandleError(fmt.Errorf("unable to upgrade: error hijacking response: %v", err))
		return nil
	}

	connWithBuf := &connWrapper{Conn: conn, bufReader: bufrw.Reader}
	spdyConn, err := NewServerConnection(connWithBuf, newStreamHandler)
	if err != nil {
		runtime.HandleError(fmt.Errorf("unable to upgrade: error creating SPDY server connection: %v", err))
		return nil
	}

	return spdyConn
}
