/*
Copyright 2022 The Kubernetes Authors.

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

package factory

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

func newCachedETCDCheck(check func() error, pollInterval time.Duration, stopCh <-chan struct{}) func() error {
	cached := &cachedETCDCheck{
		check:     check,
		cachedErr: fmt.Errorf("etcd health probe hasn't started yet"),
	}

	go wait.PollImmediateUntil(pollInterval, func() (bool, error) {
		cached.AsyncProbe()
		// always execute, until stopCh is closed
		return false, nil
	}, stopCh)

	return cached.GetCachedError
}

type cachedETCDCheck struct {
	check        func() error
	pollInterval time.Duration
	lock         sync.Mutex
	cachedErr    error
}

func (c *cachedETCDCheck) GetCachedError() error {
	c.lock.Lock()
	defer c.lock.Unlock()

	return c.cachedErr
}

func (c *cachedETCDCheck) AsyncProbe() {
	err := c.check()

	c.lock.Lock()
	defer c.lock.Unlock()
	c.cachedErr = err
}
