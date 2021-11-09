// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/*
   Copyright The containerd Authors.

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

package dialer

import (
	"context"
	"net"
	"time"

	"github.com/pkg/errors"
)

type dialResult struct {
	c   net.Conn
	err error
}

// ContextDialer returns a GRPC net.Conn connected to the provided address
func ContextDialer(ctx context.Context, address string) (net.Conn, error) {
	if deadline, ok := ctx.Deadline(); ok {
		return timeoutDialer(address, time.Until(deadline))
	}
	return timeoutDialer(address, 0)
}

// Dialer returns a GRPC net.Conn connected to the provided address
// Deprecated: use ContextDialer and grpc.WithContextDialer.
var Dialer = timeoutDialer

func timeoutDialer(address string, timeout time.Duration) (net.Conn, error) {
	var (
		stopC = make(chan struct{})
		synC  = make(chan *dialResult)
	)
	go func() {
		defer close(synC)
		for {
			select {
			case <-stopC:
				return
			default:
				c, err := dialer(address, timeout)
				if isNoent(err) {
					<-time.After(10 * time.Millisecond)
					continue
				}
				synC <- &dialResult{c, err}
				return
			}
		}
	}()
	select {
	case dr := <-synC:
		return dr.c, dr.err
	case <-time.After(timeout):
		close(stopC)
		go func() {
			dr := <-synC
			if dr != nil && dr.c != nil {
				dr.c.Close()
			}
		}()
		return nil, errors.Errorf("dial %s: timeout", address)
	}
}
