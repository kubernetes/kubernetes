/*
Copyright 2019 The Kubernetes Authors.

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

package hostport

type fakeInitSystem struct {
	fakeSystemdResolved bool
}

func NewFakeInitSystem() *fakeInitSystem {
	return &fakeInitSystem{
		fakeSystemdResolved: false,
	}
}

func NewFakeInitSystemWithSystemdResolved() *fakeInitSystem {
	return &fakeInitSystem{
		fakeSystemdResolved: true,
	}
}

func (f *fakeInitSystem) ServiceExists(service string) bool {
	if service == "systemd-resolved" {
		return f.fakeSystemdResolved
	}
	return false
}

func (f *fakeInitSystem) ServiceIsActive(service string) bool {
	if service == "systemd-resolved" {
		return f.fakeSystemdResolved
	}
	return false
}

func (f *fakeInitSystem) ServiceIsEnabled(service string) bool {
	if service == "systemd-resolved" {
		return f.fakeSystemdResolved
	}
	return false
}

func (f *fakeInitSystem) ServiceRestart(service string) error {
	return nil
}

func (f *fakeInitSystem) ServiceStart(service string) error {
	return nil
}

func (f *fakeInitSystem) ServiceStop(service string) error {
	return nil
}
