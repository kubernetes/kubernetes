/*
Copyright (c) 2020 VMware, Inc. All Rights Reserved.

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

package internal

import (
	"path"

	"github.com/vmware/govmomi/vim25/mo"
)

// InventoryPath composed of entities by Name
func InventoryPath(entities []mo.ManagedEntity) string {
	val := "/"

	for _, entity := range entities {
		// Skip root folder in building inventory path.
		if entity.Parent == nil {
			continue
		}
		val = path.Join(val, entity.Name)
	}

	return val
}
