// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package checker

import (
	"cel.dev/cel-go/common/types"
)

type mapping struct {
	mapping map[string]*types.Type
	// undo records what each add replaced while an attempt is in progress,
	// so that a failed attempt can be rolled back.
	undo      []undoEntry
	recording bool
}

type undoEntry struct {
	key  string
	prev *types.Type
	had  bool
}

func newMapping() *mapping {
	return &mapping{
		mapping: make(map[string]*types.Type),
	}
}

func (m *mapping) add(from, to *types.Type) {
	key := FormatCELType(from)
	if m.recording {
		prev, had := m.mapping[key]
		m.undo = append(m.undo, undoEntry{key, prev, had})
	}
	m.mapping[key] = to
}

// attempt runs f against m in place, rolling back what f added if it reports
// false. Attempts do not nest.
func (m *mapping) attempt(f func() bool) bool {
	m.recording = true
	ok := f()
	m.recording = false
	if !ok {
		for i := len(m.undo) - 1; i >= 0; i-- {
			e := m.undo[i]
			if e.had {
				m.mapping[e.key] = e.prev
			} else {
				delete(m.mapping, e.key)
			}
		}
	}
	m.undo = m.undo[:0]
	return ok
}

func (m *mapping) find(from *types.Type) (*types.Type, bool) {
	if r, found := m.mapping[FormatCELType(from)]; found {
		return r, found
	}
	return nil, false
}
