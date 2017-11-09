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

//FakeNetlinkHandle mock implementation of proxy NetlinkHandle
type FakeNetlinkHandle struct {
}

//NewFakeNetlinkHandle will create a new FakeNetlinkHandle
func NewFakeNetlinkHandle() *FakeNetlinkHandle {
	return &FakeNetlinkHandle{}
}

//EnsureAddressBind is a mock implementation
func (h *FakeNetlinkHandle) EnsureAddressBind(address, devName string) (exist bool, err error) {
	return false, nil
}

//UnbindAddress is a mock implementation
func (h *FakeNetlinkHandle) UnbindAddress(address, devName string) error {
	return nil
}

// EnsureDummyDevice is a mock implementation
func (h *FakeNetlinkHandle) EnsureDummyDevice(devName string) (bool, error) {
	return false, nil
}

// DeleteDummyDevice is a mock implementation
func (h *FakeNetlinkHandle) DeleteDummyDevice(devName string) error {
	return nil
}
