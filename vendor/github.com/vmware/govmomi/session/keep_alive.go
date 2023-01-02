/*
Copyright (c) 2015-2020 VMware, Inc. All Rights Reserved.

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
	"time"

	"github.com/vmware/govmomi/session/keepalive"
	"github.com/vmware/govmomi/vim25/soap"
)

// KeepAlive is a backward compatible wrapper around KeepAliveHandler.
func KeepAlive(roundTripper soap.RoundTripper, idleTime time.Duration) soap.RoundTripper {
	return KeepAliveHandler(roundTripper, idleTime, nil)
}

// KeepAliveHandler is a backward compatible wrapper around keepalive.NewHandlerSOAP.
func KeepAliveHandler(roundTripper soap.RoundTripper, idleTime time.Duration, handler func(soap.RoundTripper) error) soap.RoundTripper {
	var f func() error
	if handler != nil {
		f = func() error {
			return handler(roundTripper)
		}
	}
	return keepalive.NewHandlerSOAP(roundTripper, idleTime, f)
}
