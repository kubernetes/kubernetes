// Copyright 2019 The etcd Authors
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

package raftpb

import (
	"fmt"
	"slices"

	"google.golang.org/protobuf/proto"
)

// Equivalent returns a nil error if the inputs describe the same configuration.
// On mismatch, returns a descriptive error showing the differences.
func (cs *ConfState) Equivalent(cs2 *ConfState) error {
	if cs == nil || cs2 == nil {
		return fmt.Errorf("cannot compare ConfState: nil input (left=%v, right=%v)", cs == nil, cs2 == nil)
	}

	cs1v := proto.Clone(cs).(*ConfState)
	cs2v := proto.Clone(cs2).(*ConfState)

	s := func(sl *[]uint64) {
		slices.Sort(*sl)
	}
	for _, c := range []*ConfState{cs1v, cs2v} {
		s(&c.Voters)
		s(&c.Learners)
		s(&c.VotersOutgoing)
		s(&c.LearnersNext)

		// Treat nil AutoLeave as false.
		autoLeave := c.GetAutoLeave()
		c.AutoLeave = &autoLeave
	}

	if !proto.Equal(cs1v, cs2v) {
		return fmt.Errorf("ConfStates not equivalent after sorting:\n%+#v\n%+#v\nInputs were:\n%+#v\n%+#v", cs1v, cs2v, cs, cs2)
	}
	return nil
}
