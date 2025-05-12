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

package framework

import (
	"fmt"
	"testing"
)

type fakeData struct {
	data string
}

func (f *fakeData) Clone() StateData {
	copy := &fakeData{
		data: f.data,
	}
	return copy
}

var key StateKey = "fakedata_key"

// createCycleStateWithFakeData creates *CycleState with fakeData.
// The given data is used in stored fakeData.
func createCycleStateWithFakeData(data string, recordPluginMetrics bool) *CycleState {
	c := NewCycleState()
	c.Write(key, &fakeData{
		data: data,
	})
	c.SetRecordPluginMetrics(recordPluginMetrics)
	return c
}

// isCycleStateEqual returns whether two CycleState, which has fakeData in storage, is equal or not.
// And if they are not equal, returns message which shows why not equal.
func isCycleStateEqual(a, b *CycleState) (bool, string) {
	if a == nil && b == nil {
		return true, ""
	}
	if a == nil || b == nil {
		return false, fmt.Sprintf("one CycleState is nil, but another one is not nil. A: %v, B: %v", a, b)
	}

	if a.recordPluginMetrics != b.recordPluginMetrics {
		return false, fmt.Sprintf("CycleState A and B have a different recordPluginMetrics. A: %v, B: %v", a.recordPluginMetrics, b.recordPluginMetrics)
	}

	var msg string
	isEqual := true
	countA := 0
	a.storage.Range(func(k, v1 interface{}) bool {
		countA++
		v2, ok := b.storage.Load(k)
		if !ok {
			isEqual = false
			msg = fmt.Sprintf("CycleState B doesn't have the data which CycleState A has. key: %v, data: %v", k, v1)
			return false
		}

		typed1, ok1 := v1.(*fakeData)
		typed2, ok2 := v2.(*fakeData)
		if !ok1 || !ok2 {
			isEqual = false
			msg = fmt.Sprintf("CycleState has the data which is not type *fakeData.")
			return false
		}

		if typed1.data != typed2.data {
			isEqual = false
			msg = fmt.Sprintf("CycleState B has a different data on key %v. A: %v, B: %v", k, typed1.data, typed2.data)
			return false
		}

		return true
	})

	if !isEqual {
		return false, msg
	}

	countB := 0
	b.storage.Range(func(k, _ interface{}) bool {
		countB++
		return true
	})

	if countA != countB {
		return false, fmt.Sprintf("two Cyclestates have different numbers of data. A: %v, B: %v", countA, countB)
	}

	return true, ""
}

func TestCycleStateClone(t *testing.T) {
	tests := []struct {
		name            string
		state           *CycleState
		wantClonedState *CycleState
	}{
		{
			name:            "clone with recordPluginMetrics true",
			state:           createCycleStateWithFakeData("data", true),
			wantClonedState: createCycleStateWithFakeData("data", true),
		},
		{
			name:            "clone with recordPluginMetrics false",
			state:           createCycleStateWithFakeData("data", false),
			wantClonedState: createCycleStateWithFakeData("data", false),
		},
		{
			name:            "clone with nil CycleState",
			state:           nil,
			wantClonedState: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := tt.state
			stateCopy := state.Clone()

			if isEqual, msg := isCycleStateEqual(stateCopy, tt.wantClonedState); !isEqual {
				t.Errorf("unexpected cloned state: %v", msg)
			}

			if state == nil || stateCopy == nil {
				// not need to run the rest check in this case.
				return
			}

			stateCopy.Write(key, &fakeData{data: "modified"})
			if isEqual, _ := isCycleStateEqual(state, stateCopy); isEqual {
				t.Errorf("the change for a cloned state should not affect the original state.")
			}
		})
	}
}
