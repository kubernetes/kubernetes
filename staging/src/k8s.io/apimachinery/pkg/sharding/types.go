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

package sharding

// ShardRangeRequirement represents a single shard range requirement.
// It specifies a field path to hash and a hex range [Start, End) for filtering.
// The hash space is FNV-1a 64-bit: [0x0000000000000000, 0x10000000000000000).
// Both Start and End must be specified (no empty/unbounded values).
type ShardRangeRequirement struct {
	// Key is the field path, e.g. "object.metadata.uid"
	Key string
	// Start is the inclusive lower bound as a 0x-prefixed lowercase hex string.
	// Minimum value is "0x0000000000000000".
	Start string
	// End is the exclusive upper bound as a 0x-prefixed lowercase hex string.
	// Maximum value is "0x10000000000000000" (2^64).
	End string
}
