/*
Copyright (c) 2020 VMware, Inc. All Rights Reserved.

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

package keepalive

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
)

// handler contains the generic keep alive settings and logic
type handler struct {
	mu              sync.Mutex
	notifyStop      chan struct{}
	notifyWaitGroup sync.WaitGroup

	idle time.Duration
	send func() error
}

// NewHandlerSOAP returns a soap.RoundTripper for use with a vim25.Client
// The idle time specifies the interval in between send() requests. Defaults to 10 minutes.
// The send func is used to keep a session alive. Defaults to calling vim25 GetCurrentTime().
// The keep alive goroutine starts when a Login method is called and runs until Logout is called or send returns an error.
func NewHandlerSOAP(c soap.RoundTripper, idle time.Duration, send func() error) *HandlerSOAP {
	h := &handler{
		idle: idle,
		send: send,
	}

	if send == nil {
		h.send = func() error {
			return h.keepAliveSOAP(c)
		}
	}

	return &HandlerSOAP{h, c}
}

// NewHandlerREST returns an http.RoundTripper for use with a rest.Client
// The idle time specifies the interval in between send() requests. Defaults to 10 minutes.
// The send func is used to keep a session alive. Defaults to calling the rest.Client.Session() method
// The keep alive goroutine starts when a Login method is called and runs until Logout is called or send returns an error.
func NewHandlerREST(c *rest.Client, idle time.Duration, send func() error) *HandlerREST {
	h := &handler{
		idle: idle,
		send: send,
	}

	if send == nil {
		h.send = func() error {
			return h.keepAliveREST(c)
		}
	}

	return &HandlerREST{h, c.Transport}
}

func (h *handler) keepAliveSOAP(rt soap.RoundTripper) error {
	ctx := context.Background()
	_, err := methods.GetCurrentTime(ctx, rt)
	return err
}

func (h *handler) keepAliveREST(c *rest.Client) error {
	ctx := context.Background()

	s, err := c.Session(ctx)
	if err != nil {
		return err
	}
	if s != nil {
		return nil
	}
	return errors.New(http.StatusText(http.StatusUnauthorized))
}

// Start explicitly starts the keep alive go routine.
// For use with session cache.Client, as cached sessions may not involve Login/Logout via RoundTripper.
func (h *handler) Start() {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.notifyStop != nil {
		return
	}

	if h.idle == 0 {
		h.idle = time.Minute * 10
	}

	// This channel must be closed to terminate idle timer.
	h.notifyStop = make(chan struct{})
	h.notifyWaitGroup.Add(1)

	go func() {
		for t := time.NewTimer(h.idle); ; {
			select {
			case <-h.notifyStop:
				h.notifyWaitGroup.Done()
				t.Stop()
				return
			case <-t.C:
				if err := h.send(); err != nil {
					h.notifyWaitGroup.Done()
					h.Stop()
					return
				}
				t.Reset(h.idle)
			}
		}
	}()
}

// Stop explicitly stops the keep alive go routine.
// For use with session cache.Client, as cached sessions may not involve Login/Logout via RoundTripper.
func (h *handler) Stop() {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.notifyStop != nil {
		close(h.notifyStop)
		h.notifyWaitGroup.Wait()
		h.notifyStop = nil
	}
}

// HandlerSOAP is a keep alive implementation for use with vim25.Client
type HandlerSOAP struct {
	*handler

	roundTripper soap.RoundTripper
}

// RoundTrip implements soap.RoundTripper
func (h *HandlerSOAP) RoundTrip(ctx context.Context, req, res soap.HasFault) error {
	// Stop ticker on logout.
	switch req.(type) {
	case *methods.LogoutBody:
		h.Stop()
	}

	err := h.roundTripper.RoundTrip(ctx, req, res)
	if err != nil {
		return err
	}

	// Start ticker on login.
	switch req.(type) {
	case *methods.LoginBody, *methods.LoginExtensionByCertificateBody, *methods.LoginByTokenBody:
		h.Start()
	}

	return nil
}

// HandlerREST is a keep alive implementation for use with rest.Client
type HandlerREST struct {
	*handler

	roundTripper http.RoundTripper
}

// RoundTrip implements http.RoundTripper
func (h *HandlerREST) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Path != "/rest/com/vmware/cis/session" {
		return h.roundTripper.RoundTrip(req)
	}

	if req.Method == http.MethodDelete { // Logout
		h.Stop()
	}

	res, err := h.roundTripper.RoundTrip(req)
	if err != nil {
		return res, err
	}

	if req.Method == http.MethodPost { // Login
		h.Start()
	}

	return res, err
}
