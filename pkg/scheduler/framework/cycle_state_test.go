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

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
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

// createPluginRunningInfoWithFakeData creates running info with CycleState with fakeData.
// The given data is used in stored fakeData.
func createPluginRunningInfoWithFakeData(data string, recordPluginMetrics bool, skipPlugins ...[]string) *PluginRunningInfo {
	p := NewPluginInfo()
	p.SetRecordPluginMetrics(recordPluginMetrics)
	if len(skipPlugins) > 0 {
		p.SkipFilterPlugins = make(sets.Set[string])
		p.SkipFilterPlugins.Insert(skipPlugins[0]...)
	}
	if len(skipPlugins) > 1 {
		p.SkipScorePlugins = make(sets.Set[string])
		p.SkipScorePlugins.Insert(skipPlugins[1]...)
	}
	p.State.Write(key, &fakeData{
		data: data,
	})

	return p
}

// isCycleStateEqual returns whether two CycleState, which has fakeData in storage, is equal or not.
// And if they are not equal, returns message which shows why not equal.
func isPluginRunningInfoEqual(a, b *PluginRunningInfo) (bool, string) {
	if a == nil && b == nil {
		return true, ""
	}
	if a == nil || b == nil {
		return false, fmt.Sprintf("one PluginRunningInfo is nil, but another one is not nil. A: %v, B: %v", a, b)
	}

	if a.recordPluginMetrics != b.recordPluginMetrics {
		return false, fmt.Sprintf("PluginRunningInfo A and B have a different recordPluginMetrics. A: %v, B: %v", a.recordPluginMetrics, b.recordPluginMetrics)
	}
	if diff := cmp.Diff(a.SkipFilterPlugins, b.SkipFilterPlugins); diff != "" {
		return false, fmt.Sprintf("PluginRunningInfo A and B have different SkipFilterPlugin sets. -wanted,+got:\n%s", diff)
	}
	if diff := cmp.Diff(a.SkipScorePlugins, b.SkipScorePlugins); diff != "" {
		return false, fmt.Sprintf("PluginRunningInfo A and B have different SkipScorePlugins sets. -wanted,+got:\n%s", diff)
	}

	stateA, ok1 := a.State.(*CycleStateImpl)
	stateB, ok2 := b.State.(*CycleStateImpl)
	if !ok1 || !ok2 {
		return false, "PluginRunningInfo has cyclestate which is not type *CycleStateImpl."
	}
	var msg string
	isEqual := true
	countA := 0
	stateA.storage.Range(func(k, v1 interface{}) bool {
		countA++
		v2, ok := stateB.storage.Load(k)
		if !ok {
			isEqual = false
			msg = fmt.Sprintf("CycleState B doesn't have the data which CycleState A has. key: %v, data: %v", k, v1)
			return false
		}

		typed1, ok1 := v1.(*fakeData)
		typed2, ok2 := v2.(*fakeData)
		if !ok1 || !ok2 {
			isEqual = false
			msg = "CycleState has the data which is not type *fakeData."
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
	stateB.storage.Range(func(k, _ interface{}) bool {
		countB++
		return true
	})

	if countA != countB {
		return false, fmt.Sprintf("two Cyclestates have different numbers of data. A: %v, B: %v", countA, countB)
	}

	return true, ""
}

func TestPluginRunningInfoClone(t *testing.T) {
	tests := []struct {
		name                 string
		pluginInfo           *PluginRunningInfo
		wantClonedPluginInfo *PluginRunningInfo
	}{
		{
			name:                 "clone with recordPluginMetrics true",
			pluginInfo:           createPluginRunningInfoWithFakeData("data", true),
			wantClonedPluginInfo: createPluginRunningInfoWithFakeData("data", true),
		},
		{
			name:                 "clone with recordPluginMetrics false",
			pluginInfo:           createPluginRunningInfoWithFakeData("data", false),
			wantClonedPluginInfo: createPluginRunningInfoWithFakeData("data", false),
		},
		{
			name:                 "clone with skipScoringPlugins",
			pluginInfo:           createPluginRunningInfoWithFakeData("data", true, []string{"pl1", "pl2"}),
			wantClonedPluginInfo: createPluginRunningInfoWithFakeData("data", true, []string{"pl1", "pl2"}),
		},
		{
			name:                 "clone with skipScoringAndFilterPlugins",
			pluginInfo:           createPluginRunningInfoWithFakeData("data", false, []string{"pl1", "pl2"}, []string{"pl3"}),
			wantClonedPluginInfo: createPluginRunningInfoWithFakeData("data", false, []string{"pl1", "pl2"}, []string{"pl3"}),
		},
		{
			name:                 "clone with skipFilterPlugins",
			pluginInfo:           createPluginRunningInfoWithFakeData("data", true, []string{}, []string{"pl3", "pl4"}),
			wantClonedPluginInfo: createPluginRunningInfoWithFakeData("data", true, []string{}, []string{"pl3", "pl4"}),
		},
		{
			name:                 "clone with nil PluginInfo",
			pluginInfo:           nil,
			wantClonedPluginInfo: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pi := tt.pluginInfo
			copy := pi.Clone()

			if isEqual, msg := isPluginRunningInfoEqual(copy, tt.wantClonedPluginInfo); !isEqual {
				t.Errorf("unexpected cloned state: %v", msg)
			}

			if pi == nil || copy == nil {
				// not need to run the rest check in this case.
				return
			}

			copy.State.Write(key, &fakeData{data: "modified"})
			if isEqual, _ := isPluginRunningInfoEqual(pi, copy); isEqual {
				t.Errorf("the change for a cloned state should not affect the original state.")
			}
		})
	}
}
