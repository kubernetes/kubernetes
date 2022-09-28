/*
Package mocks provides mocks and helpers used in testing.
*/
package mocks

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
	"io"
	"net/http"
	"time"
)

// Body implements acceptable body over a string.
type Body struct {
	src           []byte
	buf           []byte
	isOpen        bool
	closeAttempts int
}

// NewBody creates a new instance of Body.
func NewBody(s string) *Body {
	return (&Body{src: []byte(s)}).reset()
}

// NewBodyWithBytes creates a new instance of Body.
func NewBodyWithBytes(b []byte) *Body {
	return (&Body{src: b}).reset()
}

// NewBodyClose creates a new instance of Body.
func NewBodyClose(s string) *Body {
	return &Body{src: []byte(s)}
}

// Read reads into the passed byte slice and returns the bytes read.
func (body *Body) Read(b []byte) (n int, err error) {
	if !body.IsOpen() {
		return 0, fmt.Errorf("ERROR: Body has been closed")
	}
	if len(body.buf) == 0 {
		return 0, io.EOF
	}
	n = copy(b, body.buf)
	body.buf = body.buf[n:]
	return n, nil
}

// Close closes the body.
func (body *Body) Close() error {
	if body.isOpen {
		body.isOpen = false
		body.closeAttempts++
	}
	return nil
}

// CloseAttempts returns the number of times Close was called.
func (body *Body) CloseAttempts() int {
	return body.closeAttempts
}

// IsOpen returns true if the Body has not been closed, false otherwise.
func (body *Body) IsOpen() bool {
	return body.isOpen
}

func (body *Body) reset() *Body {
	body.isOpen = true
	body.buf = body.src
	return body
}

// Length returns the number of bytes in the body.
func (body *Body) Length() int64 {
	if body == nil {
		return 0
	}
	return int64(len(body.src))
}

type response struct {
	r *http.Response
	e error
	d time.Duration
}

// Sender implements a simple null sender.
type Sender struct {
	attempts       int
	responses      []response
	numResponses   int
	repeatResponse []int
	err            error
	repeatError    int
	emitErrorAfter int
}

// NewSender creates a new instance of Sender.
func NewSender() *Sender {
	return &Sender{}
}

// Do accepts the passed request and, based on settings, emits a response and possible error.
func (c *Sender) Do(r *http.Request) (resp *http.Response, err error) {
	c.attempts++

	if len(c.responses) > 0 {
		resp = c.responses[0].r
		if resp != nil {
			if b, ok := resp.Body.(*Body); ok {
				b.reset()
			}
		} else {
			err = c.responses[0].e
		}
		select {
		case <-time.After(c.responses[0].d):
			// do nothing
		case <-r.Context().Done():
			err = r.Context().Err()
			return
		}
		c.repeatResponse[0]--
		if c.repeatResponse[0] == 0 {
			c.responses = c.responses[1:]
			c.repeatResponse = c.repeatResponse[1:]
		}
	} else {
		resp = NewResponse()
	}
	if resp != nil {
		resp.Request = r
	}

	if c.emitErrorAfter > 0 {
		c.emitErrorAfter--
	} else if c.err != nil {
		err = c.err
		c.repeatError--
		if c.repeatError == 0 {
			c.err = nil
		}
	}

	return
}

// AppendResponse adds the passed http.Response to the response stack.
func (c *Sender) AppendResponse(resp *http.Response) {
	c.AppendAndRepeatResponse(resp, 1)
}

// AppendResponseWithDelay adds the passed http.Response to the response stack with the specified delay.
func (c *Sender) AppendResponseWithDelay(resp *http.Response, delay time.Duration) {
	c.AppendAndRepeatResponseWithDelay(resp, delay, 1)
}

// AppendAndRepeatResponse adds the passed http.Response to the response stack along with a
// repeat count. A negative repeat count will return the response for all remaining calls to Do.
func (c *Sender) AppendAndRepeatResponse(resp *http.Response, repeat int) {
	c.appendAndRepeat(response{r: resp}, repeat)
}

// AppendAndRepeatResponseWithDelay adds the passed http.Response to the response stack with the specified
// delay along with a repeat count. A negative repeat count will return the response for all remaining calls to Do.
func (c *Sender) AppendAndRepeatResponseWithDelay(resp *http.Response, delay time.Duration, repeat int) {
	c.appendAndRepeat(response{r: resp, d: delay}, repeat)
}

// AppendError adds the passed error to the response stack.
func (c *Sender) AppendError(err error) {
	c.AppendAndRepeatError(err, 1)
}

// AppendAndRepeatError adds the passed error to the response stack along with a repeat
// count. A negative repeat count will return the response for all remaining calls to Do.
func (c *Sender) AppendAndRepeatError(err error, repeat int) {
	c.appendAndRepeat(response{e: err}, repeat)
}

func (c *Sender) appendAndRepeat(resp response, repeat int) {
	if c.responses == nil {
		c.responses = []response{resp}
		c.repeatResponse = []int{repeat}
	} else {
		c.responses = append(c.responses, resp)
		c.repeatResponse = append(c.repeatResponse, repeat)
	}
	c.numResponses++
}

// Attempts returns the number of times Do was called.
func (c *Sender) Attempts() int {
	return c.attempts
}

// SetError sets the error Do should return.
func (c *Sender) SetError(err error) {
	c.SetAndRepeatError(err, 1)
}

// SetAndRepeatError sets the error Do should return and how many calls to Do will return the error.
// A negative repeat value will return the error for all remaining calls to Do.
func (c *Sender) SetAndRepeatError(err error, repeat int) {
	c.err = err
	c.repeatError = repeat
}

// SetEmitErrorAfter sets the number of attempts to be made before errors are emitted.
func (c *Sender) SetEmitErrorAfter(ea int) {
	c.emitErrorAfter = ea
}

// NumResponses returns the number of responses that have been added to the sender.
func (c *Sender) NumResponses() int {
	return c.numResponses
}

// T is a simple testing struct.
type T struct {
	Name string `json:"name" xml:"Name"`
	Age  int    `json:"age" xml:"Age"`
}
