/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package grpc

import (
	"time"
)

// Picker picks a Conn for RPC requests.
// This is EXPERIMENTAL and Please do not implement your own Picker for now.
type Picker interface {
	// Init does initial processing for the Picker, e.g., initiate some connections.
	Init(cc *ClientConn) error
	// Pick returns the Conn to use for the upcoming RPC. It may return different
	// Conn's up to the implementation.
	Pick() (*Conn, error)
	// State returns the connectivity state of the underlying connections.
	State() ConnectivityState
	// WaitForStateChange blocks until the state changes to something other than
	// the sourceState or timeout fires on cc. It returns false if timeout fires,
	// and true otherwise.
	WaitForStateChange(timeout time.Duration, sourceState ConnectivityState) bool
	// Close closes all the Conn's owned by this Picker.
	Close() error
}

// unicastPicker is the default Picker which is used when there is no custom Picker
// specified by users. It always picks the same Conn.
type unicastPicker struct {
	conn *Conn
}

func (p *unicastPicker) Init(cc *ClientConn) error {
	c, err := NewConn(cc)
	if err != nil {
		return err
	}
	p.conn = c
	return nil
}

func (p *unicastPicker) Pick() (*Conn, error) {
	return p.conn, nil
}

func (p *unicastPicker) State() ConnectivityState {
	return p.conn.State()
}

func (p *unicastPicker) WaitForStateChange(timeout time.Duration, sourceState ConnectivityState) bool {
	return p.conn.WaitForStateChange(timeout, sourceState)
}

func (p *unicastPicker) Close() error {
	if p.conn != nil {
		return p.conn.Close()
	}
	return nil
}
