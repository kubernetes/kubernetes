/*
Copyright 2018 The Kubernetes Authors.

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

package connrotation

import (
	"context"
	"net"
	"testing"
	"time"
)

func TestCloseAll(t *testing.T) {
	closed := make(chan struct{})
	dialFn := func(ctx context.Context, network, address string) (net.Conn, error) {
		return closeOnlyConn{onClose: func() { closed <- struct{}{} }}, nil
	}
	dialer := NewDialer(dialFn)

	const numConns = 10

	// Outer loop to ensure Dialer is re-usable after CloseAll.
	for i := 0; i < 5; i++ {
		for j := 0; j < numConns; j++ {
			if _, err := dialer.Dial("", ""); err != nil {
				t.Fatal(err)
			}
		}
		dialer.CloseAll()
		for j := 0; j < numConns; j++ {
			select {
			case <-closed:
			case <-time.After(time.Second):
				t.Fatalf("iteration %d: 1s after CloseAll only %d/%d connections closed", i, j, numConns)
			}
		}
	}
}

type closeOnlyConn struct {
	net.Conn
	onClose func()
}

func (c closeOnlyConn) Close() error {
	go c.onClose()
	return nil
}
