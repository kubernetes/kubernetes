/*
Copyright The Kubernetes Authors.

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

package topologymanager

// UtilizationScore turns a pair of resource counts into the Score of a
// TopologyHint: the share of the allocatable resources on the hint's NUMA
// node(s) that is already assigned, as a percentage.
//
// It returns 0 when nothing is allocatable. 0 means "no signal" rather than
// "empty", which is why anything with allocatable resources scores at least 1
// however little of it is in use: the hint merge skips unscored contributors
// instead of averaging them in, so 0 and 1 are not neighbouring values.
//
// The resource managers share this so that CPU, memory and device scores stay
// comparable — the merge averages them with equal weight — and so the floor and
// the divide-by-zero guard cannot drift apart.
func UtilizationScore(allocatable, assigned int64) int64 {
	if allocatable <= 0 {
		return 0
	}
	return max(1, assigned*100/allocatable)
}
