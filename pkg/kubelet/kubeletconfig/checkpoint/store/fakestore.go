/*
Copyright 2017 The Kubernetes Authors.

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

package store

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
)

// so far only implements Current(), LastKnownGood(), SetCurrent(), and SetLastKnownGood()
type fakeStore struct {
	current       checkpoint.RemoteConfigSource
	lastKnownGood checkpoint.RemoteConfigSource
}

func (s *fakeStore) Initialize() error {
	return fmt.Errorf("Initialize method not supported")
}

func (s *fakeStore) Exists(uid string) (bool, error) {
	return false, fmt.Errorf("Exists method not supported")
}

func (s *fakeStore) Save(c checkpoint.Checkpoint) error {
	return fmt.Errorf("Save method not supported")
}

func (s *fakeStore) Load(uid string) (checkpoint.Checkpoint, error) {
	return nil, fmt.Errorf("Load method not supported")
}

func (s *fakeStore) CurrentModified() (time.Time, error) {
	return time.Time{}, fmt.Errorf("CurrentModified method not supported")
}

func (s *fakeStore) Current() (checkpoint.RemoteConfigSource, error) {
	return s.current, nil
}

func (s *fakeStore) LastKnownGood() (checkpoint.RemoteConfigSource, error) {
	return s.lastKnownGood, nil
}

func (s *fakeStore) SetCurrent(source checkpoint.RemoteConfigSource) error {
	s.current = source
	return nil
}

func (s *fakeStore) SetCurrentUpdated(source checkpoint.RemoteConfigSource) (bool, error) {
	return setCurrentUpdated(s, source)
}

func (s *fakeStore) SetLastKnownGood(source checkpoint.RemoteConfigSource) error {
	s.lastKnownGood = source
	return nil
}

func (s *fakeStore) Reset() (bool, error) {
	return false, fmt.Errorf("Reset method not supported")
}
