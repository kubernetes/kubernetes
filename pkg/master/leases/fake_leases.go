/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package leases

type FakeLeases struct {
	keys map[string]bool
}

var _ Leases = &FakeLeases{}

func NewFakeLeases() *FakeLeases {
	return &FakeLeases{make(map[string]bool)}
}

func (f *FakeLeases) ListLeases() ([]string, error) {
	res := make([]string, 0, len(f.keys))
	for ip := range f.keys {
		res = append(res, ip)
	}
	return res, nil
}

func (f *FakeLeases) UpdateLease(ip string) error {
	f.keys[ip] = true
	return nil
}

func (f *FakeLeases) SetLeaseTime(ttl uint64) {
}

func (f *FakeLeases) SetKeys(keys []string) {
	for _, ip := range keys {
		f.keys[ip] = false
	}
}

func (f *FakeLeases) GetUpdatedKeys() []string {
	res := []string{}
	for ip, updated := range f.keys {
		if updated {
			res = append(res, ip)
		}
	}
	return res
}
