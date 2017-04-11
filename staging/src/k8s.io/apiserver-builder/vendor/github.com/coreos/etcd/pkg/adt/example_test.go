// Copyright 2016 The etcd Authors
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

package adt_test

import (
	"fmt"

	"github.com/coreos/etcd/pkg/adt"
)

func Example() {
	ivt := &adt.IntervalTree{}

	ivt.Insert(adt.NewInt64Interval(1, 3), 123)
	ivt.Insert(adt.NewInt64Interval(9, 13), 456)
	ivt.Insert(adt.NewInt64Interval(7, 20), 789)

	rs := ivt.Stab(adt.NewInt64Point(10))
	for _, v := range rs {
		fmt.Printf("Overlapping range: %+v\n", v)
	}
	// output:
	// Overlapping range: &{Ivl:{Begin:7 End:20} Val:789}
	// Overlapping range: &{Ivl:{Begin:9 End:13} Val:456}
}
