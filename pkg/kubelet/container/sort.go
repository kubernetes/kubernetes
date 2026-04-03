/*
Copyright 2016 The Kubernetes Authors.

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

package container

import "time"

// SortContainersByAttributes defines the common ordering：
// 1. Primary sort: Descending order of Metadata.Attempt (higher attempts first)
// 2. Secondary sort: Descending order of CreatedAt (newer containers first)
// 3. Tertiary sort: Ascending order of ID (smaller IDs first)
func SortContainersByAttributes(
	attemptI, attemptJ uint32,
	createdI, createdJ int64,
	idI, idJ string,
) bool {
	if attemptI != attemptJ {
		return attemptI > attemptJ
	}
	if createdI != createdJ {
		return createdI > createdJ
	}
	return idI < idJ
}

// SortContainersStatusByAttributes defines the common ordering：
// 1. Primary sort: Descending order of RestartCount (higher attempts first)
// 2. Secondary sort: Descending order of CreatedAt (newer containers first)
// 3. Tertiary sort: Ascending order of ID (smaller IDs first)
func SortContainersStatusByAttributes(
	restartCountI, restartCountJ int,
	createdAtI, createdAtJ time.Time,
	idI, idJ string,
) bool {
	if restartCountI != restartCountJ {
		return restartCountI > restartCountJ
	}
	if !createdAtI.Equal(createdAtJ) {
		return createdAtI.After(createdAtJ)
	}
	return idI < idJ
}
