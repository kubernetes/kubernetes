// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nodes

import (
	"fmt"
	"testing"

	"github.com/coreos/fleet/machine"
	"github.com/coreos/fleet/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeFleetClientApi struct {
	f func() ([]machine.MachineState, error)
}

func (self *fakeFleetClientApi) Machines() ([]machine.MachineState, error) {
	return self.f()
}

func (self *fakeFleetClientApi) Unit(string) (*schema.Unit, error) {
	return nil, nil
}
func (self *fakeFleetClientApi) Units() ([]*schema.Unit, error) {
	return nil, nil
}
func (self *fakeFleetClientApi) UnitStates() ([]*schema.UnitState, error) {
	return nil, nil
}
func (self *fakeFleetClientApi) SetUnitTargetState(name, target string) error {
	return nil
}
func (self *fakeFleetClientApi) CreateUnit(*schema.Unit) error {
	return nil
}
func (self *fakeFleetClientApi) DestroyUnit(string) error {
	return nil
}

func TestSuccessCase(t *testing.T) {
	fakeClient := &fakeFleetClientApi{}
	fakeClient.f = func() ([]machine.MachineState, error) {
		return []machine.MachineState{
			{ID: "a", PublicIP: "1.2.3.4"},
			{ID: "b", PublicIP: "1.2.3.5"},
		}, nil
	}
	nodesApi := &fleetNodes{client: fakeClient}
	nodeList, err := nodesApi.List()
	require.NoError(t, err)
	assert.Len(t, nodeList.Items, 2)
	assert.Equal(t, nodeList.Items["a"], Info{"1.2.3.4", "1.2.3.4", "", 0, 0})
	assert.Equal(t, nodeList.Items["b"], Info{"1.2.3.5", "1.2.3.5", "", 0, 0})
}

func TestFailureCase(t *testing.T) {
	fakeClient := &fakeFleetClientApi{}
	fakeClient.f = func() ([]machine.MachineState, error) {
		return nil, fmt.Errorf("test error")
	}
	nodesApi := &fleetNodes{client: fakeClient}
	_, err := nodesApi.List()
	require.EqualError(t, err, "test error")
}
