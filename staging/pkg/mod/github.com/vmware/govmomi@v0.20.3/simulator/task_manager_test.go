/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"testing"

	"github.com/vmware/govmomi/simulator/esx"
)

func TestTaskManagerRecent(t *testing.T) {
	m := ESX()
	m.Datastore = 0
	m.Machine = 0

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	recentTaskMax = 5

	tm := Map.Get(*esx.ServiceContent.TaskManager).(*TaskManager)
	tm.RecentTask = nil

	for i := 0; i < recentTaskMax+2; i++ {
		CreateTask(esx.RootFolder, "noop", nil)

		if len(tm.RecentTask) > recentTaskMax {
			t.Errorf("too many tasks %d > %d", len(tm.RecentTask), recentTaskMax)
		}
	}
}
