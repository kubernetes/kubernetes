/*
Copyright 2015 The Kubernetes Authors.

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

package service

import (
	"testing"
	"time"

	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resources"

	"github.com/stretchr/testify/assert"
)

type fakeSchedulerProcess struct {
	doneFunc     func() <-chan struct{}
	failoverFunc func() <-chan struct{}
}

func (self *fakeSchedulerProcess) Terminal() <-chan struct{} {
	if self == nil || self.doneFunc == nil {
		return nil
	}
	return self.doneFunc()
}

func (self *fakeSchedulerProcess) Failover() <-chan struct{} {
	if self == nil || self.failoverFunc == nil {
		return nil
	}
	return self.failoverFunc()
}

func (self *fakeSchedulerProcess) End() <-chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

func Test_awaitFailoverDone(t *testing.T) {
	done := make(chan struct{})
	p := &fakeSchedulerProcess{
		doneFunc: func() <-chan struct{} { return done },
	}
	ss := &SchedulerServer{}
	failoverHandlerCalled := false
	failoverFailedHandler := func() error {
		failoverHandlerCalled = true
		return nil
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- ss.awaitFailover(p, failoverFailedHandler)
	}()
	close(done)
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("timed out waiting for failover")
	}
	if failoverHandlerCalled {
		t.Fatalf("unexpected call to failover handler")
	}
}

func Test_awaitFailoverDoneFailover(t *testing.T) {
	ch := make(chan struct{})
	p := &fakeSchedulerProcess{
		doneFunc:     func() <-chan struct{} { return ch },
		failoverFunc: func() <-chan struct{} { return ch },
	}
	ss := &SchedulerServer{}
	failoverHandlerCalled := false
	failoverFailedHandler := func() error {
		failoverHandlerCalled = true
		return nil
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- ss.awaitFailover(p, failoverFailedHandler)
	}()
	close(ch)
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("timed out waiting for failover")
	}
	if !failoverHandlerCalled {
		t.Fatalf("expected call to failover handler")
	}
}

func Test_DefaultResourceLimits(t *testing.T) {
	assert := assert.New(t)

	s := NewSchedulerServer()
	assert.Equal(s.defaultContainerCPULimit, resources.DefaultDefaultContainerCPULimit)
	assert.Equal(s.defaultContainerMemLimit, resources.DefaultDefaultContainerMemLimit)
}
