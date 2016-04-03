// Copyright 2015 CoreOS, Inc.
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

package etcdserver

import "testing"

func TestConsistentIndex(t *testing.T) {
	var i consistentIndex
	i.setConsistentIndex(10)
	if g := i.ConsistentIndex(); g != 10 {
		t.Errorf("value = %d, want 10", g)
	}
}
