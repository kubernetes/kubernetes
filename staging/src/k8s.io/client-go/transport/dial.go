/*
Copyright 2017 The Kubernetes Authors.

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

package transport

import (
	"net"
	"time"

	"github.com/golang/glog"
	"github.com/obeattie/tcp-failfast"
)

type dialFunc func(network, addr string) (net.Conn, error)

// failFastDial wraps a dial function and sets up a TCP "user timeout" on the
// connection. This is useful because we want to detect dead connections to
// the apiserver quickly, and default kernel parameters mean this usually takes
// 15+ minutes.
func failFastDial(d dialFunc, timeout time.Duration) dialFunc {
	return func(network, addr string) (net.Conn, error) {
		conn, err := d(network, addr)
		if err != nil {
			return conn, err
		}

		if tcp, ok := conn.(*net.TCPConn); ok {
			tcpErr := tcpfailfast.FailFastTCP(tcp, timeout)
			switch tcpErr {
			case nil:
				glog.V(2).Infof("Enabled TCP failfast on connection to %s:%s. Connections will be terminated after %v of unacknowledged transmissions.", network, addr, timeout)
			case tcpfailfast.ErrUnsupported:
				glog.Warning("TCP failfast is not supported on this platform. It may take a long time (>15 minutes) to detect connection drops, depending on kernel config.")
			default:
				// It would be possible to return (conn, tcpErr) here but callers do
				// not generally expect to have to call Close() when dial errors.
				conn.Close()
				return nil, tcpErr
			}
		}

		return conn, err
	}
}
