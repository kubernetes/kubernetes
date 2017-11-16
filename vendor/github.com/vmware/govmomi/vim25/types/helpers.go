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

package types

import "strings"

func NewBool(v bool) *bool {
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
