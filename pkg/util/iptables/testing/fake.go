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

package testing

import "k8s.io/kubernetes/pkg/util/iptables"

// no-op implementation of iptables Interface
type fake struct{}

func NewFake() *fake {
	return &fake{}
}

func (*fake) GetVersion() (string, error) {
	return "0.0.0", nil
}

func (*fake) EnsureChain(table iptables.Table, chain iptables.Chain) (bool, error) {
	return true, nil
}

func (*fake) FlushChain(table iptables.Table, chain iptables.Chain) error {
	return nil
}

func (*fake) DeleteChain(table iptables.Table, chain iptables.Chain) error {
	return nil
}

func (*fake) EnsureRule(position iptables.RulePosition, table iptables.Table, chain iptables.Chain, args ...string) (bool, error) {
	return true, nil
}

func (*fake) DeleteRule(table iptables.Table, chain iptables.Chain, args ...string) error {
	return nil
}

func (*fake) IsIpv6() bool {
	return false
}

func (*fake) Save(table iptables.Table) ([]byte, error) {
	return make([]byte, 0), nil
}

func (*fake) SaveAll() ([]byte, error) {
	return make([]byte, 0), nil
}

func (*fake) Restore(table iptables.Table, data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	return nil
}

func (*fake) RestoreAll(data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	return nil
}
func (*fake) AddReloadFunc(reloadFunc func()) {}

func (*fake) Destroy() {}

var _ = iptables.Interface(&fake{})
