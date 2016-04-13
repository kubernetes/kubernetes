// Copyright 2015 Google Inc. All Rights Reserved.
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

package util

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/util"
)

// ClientInitializer manages client initialization for sinks.
type ClientInitializer interface {
	// Done returns true iff client has been initialized successfully.
	Done() bool
}

type clientInitializerImpl struct {
	name             string
	clientConfigured atomic.Value
	sync.Mutex
	initializer func() error
	ping        func() error
	frequency   time.Duration
}

func (cii *clientInitializerImpl) setClientConfigured(val bool) {
	cii.Lock()
	defer cii.Unlock()
	cii.clientConfigured.Store(val)

}
func (cii *clientInitializerImpl) setup() {
	if cii.Done() {
		// Check if client is healthy.
		if err := cii.ping(); err == nil {
			return
		}
		cii.setClientConfigured(false)
	}
	err := cii.initializer()
	if err != nil {
		glog.Errorf("Failed to initialize client %q- %v", cii.name, err)
		return
	}
	cii.setClientConfigured(true)
	glog.V(3).Infof("Client %q initalized successfully", cii.name)

}

func (cii *clientInitializerImpl) Done() bool {
	return cii.clientConfigured.Load().(bool)
}

// NewClientInitializer returns a ClientInitializer object that will attempt to initialize
// sink client periodically based on frequency, until successful.
func NewClientInitializer(name string, initializer func() error, ping func() error, frequency time.Duration) ClientInitializer {
	ret := &clientInitializerImpl{
		name:        name,
		initializer: initializer,
		ping:        ping,
	}
	ret.clientConfigured.Store(false)
	// Try once now.
	ret.setup()
	go util.Until(ret.setup, frequency, util.NeverStop)
	return ret
}
