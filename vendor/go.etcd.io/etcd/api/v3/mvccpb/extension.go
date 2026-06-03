// Copyright 2026 The etcd Authors
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

package mvccpb

// IsCreate returns true if the event tells that the key is newly created.
func (e *Event) IsCreate() bool {
	return e.GetType() == PUT && e.GetKv().GetCreateRevision() == e.GetKv().GetModRevision()
}

// IsModify returns true if the event tells that a new value is put on existing key.
func (e *Event) IsModify() bool {
	return e.GetType() == PUT && e.GetKv().GetCreateRevision() != e.GetKv().GetModRevision()
}
