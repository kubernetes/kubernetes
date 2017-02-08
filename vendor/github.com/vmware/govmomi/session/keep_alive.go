/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package session

import (
	"context"
	"sync"
	"time"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
)

type keepAlive struct {
	sync.Mutex

	roundTripper    soap.RoundTripper
	idleTime        time.Duration
	notifyRequest   chan struct{}
	notifyStop      chan struct{}
	notifyWaitGroup sync.WaitGroup

	// keepAlive executes a request in the background with the purpose of
	// keeping the session active. The response for this request is discarded.
	keepAlive func(soap.RoundTripper) error
}

func defaultKeepAlive(roundTripper soap.RoundTripper) error {
	_, _ = methods.GetCurrentTime(context.Background(), roundTripper)
	return nil
}

// KeepAlive wraps the specified soap.RoundTripper and executes a meaningless
// API request in the background after the RoundTripper has been idle for the
// specified amount of idle time. The keep alive process only starts once a
// user logs in and runs until the user logs out again.
func KeepAlive(roundTripper soap.RoundTripper, idleTime time.Duration) soap.RoundTripper {
	return KeepAliveHandler(roundTripper, idleTime, defaultKeepAlive)
}

// KeepAliveHandler works as KeepAlive() does, but the handler param can decide how to handle errors.
// For example, if connectivity to ESX/VC is down long enough for a session to expire, a handler can choose to
// Login() on a types.NotAuthenticated error.  If handler returns non-nil, the keep alive go routine will be stopped.
func KeepAliveHandler(roundTripper soap.RoundTripper, idleTime time.Duration, handler func(soap.RoundTripper) error) soap.RoundTripper {
	k := &keepAlive{
		roundTripper:  roundTripper,
		idleTime:      idleTime,
		notifyRequest: make(chan struct{}),
	}

	k.keepAlive = handler

	return k
}

func (k *keepAlive) start() {
	k.Lock()
	defer k.Unlock()

	if k.notifyStop != nil {
		return
	}

	// This channel must be closed to terminate idle timer.
	k.notifyStop = make(chan struct{})
	k.notifyWaitGroup.Add(1)

	go func() {
		defer k.notifyWaitGroup.Done()

		for t := time.NewTimer(k.idleTime); ; {
			select {
			case <-k.notifyStop:
				return
			case <-k.notifyRequest:
				t.Reset(k.idleTime)
			case <-t.C:
				if err := k.keepAlive(k.roundTripper); err != nil {
					k.stop()
				}
				t = time.NewTimer(k.idleTime)
			}
		}
	}()
}

func (k *keepAlive) stop() {
	k.Lock()
	defer k.Unlock()

	if k.notifyStop != nil {
		close(k.notifyStop)
		k.notifyWaitGroup.Wait()
		k.notifyStop = nil
	}
}

func (k *keepAlive) RoundTrip(ctx context.Context, req, res soap.HasFault) error {
	err := k.roundTripper.RoundTrip(ctx, req, res)
	if err != nil {
		return err
	}

	// Start ticker on login, stop ticker on logout.
	switch req.(type) {
	case *methods.LoginBody, *methods.LoginExtensionByCertificateBody:
		k.start()
	case *methods.LogoutBody:
		k.stop()
	}

	return nil
}
