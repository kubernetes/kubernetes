// Copyright 2015 The rkt Authors
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

//+build linux

package cgroup

import (
	"io"
	"reflect"
	"strings"
	"testing"
)

func TestParseCgroups(t *testing.T) {
	cg1 := `#subsys_name	hierarchy	num_cgroups	enabled
cpuset	2	1	1
cpu	3	1	1
cpuacct	3	1	1
blkio	4	1	1
memory	6	1	1
devices	7	47	1
freezer	8	1	1
net_cls	5	1	1`

	cg2 := `#subsys_name	hierarchy	num_cgroups	enabled
cpuset	8	441	1
cpu	4	31	1
cpuacct	4	31	1
blkio	2	13	1
memory	0	1	0
devices	3	88	1
freezer	7	432	1
net_cls	6	432	1
perf_event	5	432	1
net_prio	6	432	1`

	cg3 := `#subsys_name	hierarchy	num_cgroups	enabled
cpuset	1	441	1
cpu	4	31	1
cpuacct	4	31	0
blkio	2	13	1
memory	0	1	0
devices	3	88	1
freezer	7	432	1
net_cls	6	432	1
perf_event	5	432	0
net_prio	6	432	1`

	tests := []struct {
		input  io.Reader
		output map[int][]string
	}{
		{
			input: strings.NewReader(cg1),
			output: map[int][]string{
				2: []string{"cpuset"},
				3: []string{"cpu", "cpuacct"},
				4: []string{"blkio"},
				6: []string{"memory"},
				7: []string{"devices"},
				8: []string{"freezer"},
				5: []string{"net_cls"},
			},
		},
		{
			input: strings.NewReader(cg2),
			output: map[int][]string{
				8: []string{"cpuset"},
				4: []string{"cpu", "cpuacct"},
				2: []string{"blkio"},
				3: []string{"devices"},
				7: []string{"freezer"},
				6: []string{"net_cls", "net_prio"},
				5: []string{"perf_event"},
			},
		},
		{
			input: strings.NewReader(cg3),
			output: map[int][]string{
				1: []string{"cpuset"},
				4: []string{"cpu"},
				2: []string{"blkio"},
				3: []string{"devices"},
				7: []string{"freezer"},
				6: []string{"net_cls", "net_prio"},
			},
		},
	}

	for i, tt := range tests {
		o, err := parseCgroups(tt.input)
		if err != nil {
			t.Errorf("#%d: unexpected error `%v`", i, err)
		}
		eq := reflect.DeepEqual(o, tt.output)
		if !eq {
			t.Errorf("#%d: expected `%v` got `%v`", i, tt.output, o)
		}
	}
}
