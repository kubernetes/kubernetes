/*
Copyright 2023 The Kubernetes Authors.

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

package server

import (
	"testing"
	"time"

	clocktesting "k8s.io/utils/clock/testing"
)

func TestLifecycleSignal(t *testing.T) {
	signalName := "mysignal"
	signaledAt := time.Now()
	clock := clocktesting.NewFakeClock(signaledAt)
	s := newNamedChannelWrapper(signalName, clock)

	if s.Name() != signalName {
		t.Errorf("expected signal name to match: %q, but got: %q", signalName, s.Name())
	}
	if at := s.SignaledAt(); at != nil {
		t.Errorf("expected SignaledAt to return nil, but got: %v", *at)
	}
	select {
	case <-s.Signaled():
		t.Errorf("expected the lifecycle event to not be signaled initially")
	default:
	}

	s.Signal()

	if at := s.SignaledAt(); at == nil || !at.Equal(signaledAt) {
		t.Errorf("expected SignaledAt to return %v, but got: %v", signaledAt, at)
	}
	select {
	case <-s.Signaled():
	default:
		t.Errorf("expected the lifecycle event to be signaled")
	}
}
