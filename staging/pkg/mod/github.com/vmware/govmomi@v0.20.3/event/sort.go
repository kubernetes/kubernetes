/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package event

import (
	"sort"

	"github.com/vmware/govmomi/vim25/types"
)

// Sort events in ascending order base on Key
// From the EventHistoryCollector.latestPage sdk docs:
//   The "oldest event" is the one with the smallest key (event ID).
//   The events in the returned page are unordered.
func Sort(events []types.BaseEvent) {
	sort.Sort(baseEvent(events))
}

type baseEvent []types.BaseEvent

func (d baseEvent) Len() int {
	return len(d)
}

func (d baseEvent) Less(i, j int) bool {
	return d[i].GetEvent().Key < d[j].GetEvent().Key
}

func (d baseEvent) Swap(i, j int) {
	d[i], d[j] = d[j], d[i]
}
