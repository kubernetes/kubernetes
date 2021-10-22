// Copyright 2019, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmp_test

import (
	"fmt"
	"strings"

	"github.com/google/go-cmp/cmp"
)

// DiffReporter is a simple custom reporter that only records differences
// detected during comparison.
type DiffReporter struct {
	path  cmp.Path
	diffs []string
}

func (r *DiffReporter) PushStep(ps cmp.PathStep) {
	r.path = append(r.path, ps)
}

func (r *DiffReporter) Report(rs cmp.Result) {
	if !rs.Equal() {
		vx, vy := r.path.Last().Values()
		r.diffs = append(r.diffs, fmt.Sprintf("%#v:\n\t-: %+v\n\t+: %+v\n", r.path, vx, vy))
	}
}

func (r *DiffReporter) PopStep() {
	r.path = r.path[:len(r.path)-1]
}

func (r *DiffReporter) String() string {
	return strings.Join(r.diffs, "\n")
}

func ExampleReporter() {
	x, y := MakeGatewayInfo()

	var r DiffReporter
	cmp.Equal(x, y, cmp.Reporter(&r))
	fmt.Print(r.String())

	// Output:
	// {cmp_test.Gateway}.IPAddress:
	// 	-: 192.168.0.1
	// 	+: 192.168.0.2
	//
	// {cmp_test.Gateway}.Clients[4].IPAddress:
	// 	-: 192.168.0.219
	// 	+: 192.168.0.221
	//
	// {cmp_test.Gateway}.Clients[5->?]:
	// 	-: {Hostname:americano IPAddress:192.168.0.188 LastSeen:2009-11-10 23:03:05 +0000 UTC}
	// 	+: <invalid reflect.Value>
}
