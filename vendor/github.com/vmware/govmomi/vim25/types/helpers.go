/*
Copyright (c) 2015-2017 VMware, Inc. All Rights Reserved.

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

package types

import (
	"reflect"
	"strings"
	"time"
)

func NewBool(v bool) *bool {
	return &v
}

func NewInt32(v int32) *int32 {
	return &v
}

func NewInt64(v int64) *int64 {
	return &v
}

func NewTime(v time.Time) *time.Time {
	return &v
}

func NewReference(r ManagedObjectReference) *ManagedObjectReference {
	return &r
}

func (r ManagedObjectReference) Reference() ManagedObjectReference {
	return r
}

func (r ManagedObjectReference) String() string {
	return strings.Join([]string{r.Type, r.Value}, ":")
}

func (r *ManagedObjectReference) FromString(o string) bool {
	s := strings.SplitN(o, ":", 2)

	if len(s) < 2 {
		return false
	}

	r.Type = s[0]
	r.Value = s[1]

	return true
}

func (c *PerfCounterInfo) Name() string {
	return c.GroupInfo.GetElementDescription().Key + "." + c.NameInfo.GetElementDescription().Key + "." + string(c.RollupType)
}

func defaultResourceAllocationInfo() ResourceAllocationInfo {
	return ResourceAllocationInfo{
		Reservation:           NewInt64(0),
		ExpandableReservation: NewBool(true),
		Limit: NewInt64(-1),
		Shares: &SharesInfo{
			Level: SharesLevelNormal,
		},
	}
}

// DefaultResourceConfigSpec returns a ResourceConfigSpec populated with the same default field values as vCenter.
// Note that the wsdl marks these fields as optional, but they are required to be set when creating a resource pool.
// They are only optional when updating a resource pool.
func DefaultResourceConfigSpec() ResourceConfigSpec {
	return ResourceConfigSpec{
		CpuAllocation:    defaultResourceAllocationInfo(),
		MemoryAllocation: defaultResourceAllocationInfo(),
	}
}

func init() {
	// Known 6.5 issue where this event type is sent even though it is internal.
	// This workaround allows us to unmarshal and avoid NPEs.
	t["HostSubSpecificationUpdateEvent"] = reflect.TypeOf((*HostEvent)(nil)).Elem()
}
