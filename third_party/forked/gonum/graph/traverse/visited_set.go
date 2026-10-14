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

package traverse

// visitedSet records the IDs of the nodes a traversal has already seen.
//
// It exists because gonum's intsets.Sparse chains fixed-size blocks in a linked
// list that Has and Insert walk from the head. IDs handed out by the simple
// graph spread over the whole ID space and neighbors arrive in arbitrary order,
// so membership checks walked most of that list and a traversal of n nodes cost
// O(n^2).
//
// The zero value is ready to use.
type visitedSet struct {
	// ids holds the visited IDs while they are few enough that scanning them is
	// cheaper than allocating a map. It is emptied once set takes over.
	ids []int
	// set takes over once ids reaches visitedSetMaxIDs, and from then on holds
	// every visited ID.
	set map[int]struct{}
}

// visitedSetMaxIDs is the number of IDs kept in the linear list before
// switching to the map. Most authorization traversals never reach it.
const visitedSetMaxIDs = 16

func (s *visitedSet) has(id int) bool {
	if s.set != nil {
		_, visited := s.set[id]
		return visited
	}
	for _, visited := range s.ids {
		if visited == id {
			return true
		}
	}
	return false
}

func (s *visitedSet) insert(id int) {
	if s.set == nil {
		if len(s.ids) < visitedSetMaxIDs {
			if s.ids == nil {
				s.ids = make([]int, 0, visitedSetMaxIDs)
			}
			s.ids = append(s.ids, id)
			return
		}
		// Move the list into the map so that lookups past this point are a
		// single probe rather than a scan of the list plus a probe.
		s.set = make(map[int]struct{}, 2*visitedSetMaxIDs)
		for _, visited := range s.ids {
			s.set[visited] = struct{}{}
		}
		s.ids = s.ids[:0]
	}
	s.set[id] = struct{}{}
}

func (s *visitedSet) clear() {
	s.ids = s.ids[:0]
	clear(s.set)
}
