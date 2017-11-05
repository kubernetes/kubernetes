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

package testing

import (
	"k8s.io/kubernetes/pkg/util/ipset"
)

// FakeIPSet is a no-op implementation of ipset Interface
type FakeIPSet struct {
	Lines []byte
}

// NewFake create a new fake ipset interface.
func NewFake() *FakeIPSet {
	return &FakeIPSet{}
}

// GetVersion is part of interface.
func (*FakeIPSet) GetVersion() (string, error) {
	return "0.0", nil
}

// FlushSet is part of interface.
func (*FakeIPSet) FlushSet(set string) error {
	return nil
}

// DestroySet is part of interface.
func (*FakeIPSet) DestroySet(set string) error {
	return nil
}

// DestroyAllSets is part of interface.
func (*FakeIPSet) DestroyAllSets() error {
	return nil
}

// CreateSet is part of interface.
func (*FakeIPSet) CreateSet(set *ipset.IPSet, ignoreExistErr bool) error {
	return nil
}

// AddEntry is part of interface.
func (*FakeIPSet) AddEntry(entry string, set string, ignoreExistErr bool) error {
	return nil
}

// DelEntry is part of interface.
func (*FakeIPSet) DelEntry(entry string, set string) error {
	return nil
}

// TestEntry is part of interface.
func (*FakeIPSet) TestEntry(entry string, set string) (bool, error) {
	return true, nil
}

// ListEntries is part of interface.
func (*FakeIPSet) ListEntries(set string) ([]string, error) {
	return nil, nil
}

// ListSets is part of interface.
func (*FakeIPSet) ListSets() ([]string, error) {
	return nil, nil
}

var _ = ipset.Interface(&FakeIPSet{})
