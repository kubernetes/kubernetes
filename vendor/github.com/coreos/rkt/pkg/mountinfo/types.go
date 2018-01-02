// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mountinfo

// Mount contains information about a single mountpoint entry
type Mount struct {
	ID         int
	Parent     int
	Major      int
	Minor      int
	Root       string
	MountPoint string
	Opts       map[string]struct{}
}

// FilterFunc is a functional type which returns true if the given Mount should be filtered, else false.
type FilterFunc func(m *Mount) bool

// NeedsRemountPrivate checks if this mountPoint needs to be remounted
// as private, in order for children to be properly unmounted without
// leaking to parents.
func (m *Mount) NeedsRemountPrivate() bool {
	for _, key := range []string{
		"shared",
		"master",
	} {
		if _, needsRemount := m.Opts[key]; needsRemount {
			return true
		}
	}
	return false
}

// Mounts represents a sortable set of mountpoint entries.
// It implements sort.Interface according to unmount order (children first).
type Mounts []*Mount

// Filter returns a filtered copy of Mounts
func (ms Mounts) Filter(f FilterFunc) Mounts {
	filtered := make([]*Mount, 0, len(ms))

	for _, m := range ms {
		if f(m) {
			filtered = append(filtered, m)
		}
	}

	return Mounts(filtered)
}

// Less ensures that mounts are sorted in an order we can unmount; descendant before ancestor.
// The requirement of transitivity for Less has to be fulfilled otherwise the sort algorithm will fail.
func (ms Mounts) Less(i, j int) (result bool) { return ms.mountDepth(i) >= ms.mountDepth(j) }

func (ms Mounts) Len() int { return len(ms) }

func (ms Mounts) Swap(i, j int) { ms[i], ms[j] = ms[j], ms[i] }

// mountDepth determines and returns the number of ancestors of the mount at index i
func (ms Mounts) mountDepth(i int) int {
	ancestorCount := 0
	current := ms[i]
	for found := true; found; {
		found = false
		for _, mnt := range ms {
			if mnt.ID == current.Parent {
				ancestorCount++
				current = mnt
				found = true
				break
			}
		}
	}
	return ancestorCount
}
