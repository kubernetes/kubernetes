/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package common

import (
	"bytes"
	compute "code.google.com/p/google-api-go-client/compute/v1"
	"encoding/json"
	"sync"
)

// returns json, hash pair
type ConfigInterface interface {
	Load() ([]*compute.Firewall, []byte, int)
	Store(firewallRules []*compute.Firewall) (bool, error)
}

type FirewallConfig struct {
	serialized    []byte
	firewallRules []*compute.Firewall
	updateCount   int
	mutex         *sync.Mutex
}

func NewFirewallConfig() ConfigInterface {
	return &FirewallConfig{
		serialized:  []byte{},
		updateCount: 0,
		mutex:       &sync.Mutex{},
	}
}

func (fc *FirewallConfig) Load() ([]*compute.Firewall, []byte, int) {
	fc.mutex.Lock()
	defer fc.mutex.Unlock()
	return fc.firewallRules, fc.serialized, fc.updateCount
}

func (fc *FirewallConfig) Store(firewallRules []*compute.Firewall) (bool, error) {
	fc.mutex.Lock()
	defer fc.mutex.Unlock()
	serialized, err := json.Marshal(firewallRules)
	if err != nil {
		return false, err
	}
	if fc.updateCount == 0 || bytes.Compare(serialized, fc.serialized) != 0 {
		fc.firewallRules = firewallRules
		fc.serialized = serialized
		fc.updateCount = fc.updateCount + 1
		return true, nil
	}
	return false, nil
}
