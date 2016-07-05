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

package framework

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Check that same slave is only added once.
func TestSlaveStorage_Register(t *testing.T) {
	assert := assert.New(t)

	slaveStorage := newSlaveRegistry()
	assert.Equal(0, len(slaveStorage.hostNames))

	slaveId := "slave1"
	slaveHostname := "slave1Hostname"
	slaveStorage.Register(slaveId, slaveHostname)
	assert.Equal(1, len(slaveStorage.SlaveIDs()))

	slaveStorage.Register(slaveId, slaveHostname)
	assert.Equal(1, len(slaveStorage.SlaveIDs()))
}

// Check that GetHostName returns empty hostname for nonexisting slave.
func TestSlaveStorage_HostName(t *testing.T) {
	assert := assert.New(t)

	slaveStorage := newSlaveRegistry()
	assert.Equal(0, len(slaveStorage.hostNames))

	slaveId := "slave1"
	slaveHostname := "slave1Hostname"

	h := slaveStorage.HostName(slaveId)
	assert.Equal(h, "")

	slaveStorage.Register(slaveId, slaveHostname)
	assert.Equal(1, len(slaveStorage.SlaveIDs()))

	h = slaveStorage.HostName(slaveId)
	assert.Equal(h, slaveHostname)
}

// Check that GetSlaveIds returns array with all slaveIds.
func TestSlaveStorage_SlaveIds(t *testing.T) {
	assert := assert.New(t)

	slaveStorage := newSlaveRegistry()
	assert.Equal(0, len(slaveStorage.hostNames))

	slaveId := "1"
	slaveHostname := "hn1"
	slaveStorage.Register(slaveId, slaveHostname)
	assert.Equal(1, len(slaveStorage.SlaveIDs()))

	slaveId = "2"
	slaveHostname = "hn2"
	slaveStorage.Register(slaveId, slaveHostname)
	assert.Equal(2, len(slaveStorage.SlaveIDs()))

	slaveIds := slaveStorage.SlaveIDs()

	slaveIdsMap := make(map[string]bool, len(slaveIds))
	for _, s := range slaveIds {
		slaveIdsMap[s] = true
	}

	_, ok := slaveIdsMap["1"]
	assert.Equal(ok, true)

	_, ok = slaveIdsMap["2"]
	assert.Equal(ok, true)
}
