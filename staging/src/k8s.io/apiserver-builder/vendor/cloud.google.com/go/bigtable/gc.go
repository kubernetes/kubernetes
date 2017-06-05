/*
Copyright 2015 Google Inc. All Rights Reserved.

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

package bigtable

import (
	"fmt"
	"strings"
	"time"

	durpb "github.com/golang/protobuf/ptypes/duration"
	bttdpb "google.golang.org/genproto/googleapis/bigtable/admin/v2"
)

// A GCPolicy represents a rule that determines which cells are eligible for garbage collection.
type GCPolicy interface {
	String() string
	proto() *bttdpb.GcRule
}

// IntersectionPolicy returns a GC policy that only applies when all its sub-policies apply.
func IntersectionPolicy(sub ...GCPolicy) GCPolicy { return intersectionPolicy{sub} }

type intersectionPolicy struct {
	sub []GCPolicy
}

func (ip intersectionPolicy) String() string {
	var ss []string
	for _, sp := range ip.sub {
		ss = append(ss, sp.String())
	}
	return "(" + strings.Join(ss, " && ") + ")"
}

func (ip intersectionPolicy) proto() *bttdpb.GcRule {
	inter := &bttdpb.GcRule_Intersection{}
	for _, sp := range ip.sub {
		inter.Rules = append(inter.Rules, sp.proto())
	}
	return &bttdpb.GcRule{
		Rule: &bttdpb.GcRule_Intersection_{inter},
	}
}

// UnionPolicy returns a GC policy that applies when any of its sub-policies apply.
func UnionPolicy(sub ...GCPolicy) GCPolicy { return unionPolicy{sub} }

type unionPolicy struct {
	sub []GCPolicy
}

func (up unionPolicy) String() string {
	var ss []string
	for _, sp := range up.sub {
		ss = append(ss, sp.String())
	}
	return "(" + strings.Join(ss, " || ") + ")"
}

func (up unionPolicy) proto() *bttdpb.GcRule {
	union := &bttdpb.GcRule_Union{}
	for _, sp := range up.sub {
		union.Rules = append(union.Rules, sp.proto())
	}
	return &bttdpb.GcRule{
		Rule: &bttdpb.GcRule_Union_{union},
	}
}

// MaxVersionsPolicy returns a GC policy that applies to all versions of a cell
// except for the most recent n.
func MaxVersionsPolicy(n int) GCPolicy { return maxVersionsPolicy(n) }

type maxVersionsPolicy int

func (mvp maxVersionsPolicy) String() string { return fmt.Sprintf("versions() > %d", int(mvp)) }

func (mvp maxVersionsPolicy) proto() *bttdpb.GcRule {
	return &bttdpb.GcRule{Rule: &bttdpb.GcRule_MaxNumVersions{int32(mvp)}}
}

// MaxAgePolicy returns a GC policy that applies to all cells
// older than the given age.
func MaxAgePolicy(d time.Duration) GCPolicy { return maxAgePolicy(d) }

type maxAgePolicy time.Duration

var units = []struct {
	d      time.Duration
	suffix string
}{
	{24 * time.Hour, "d"},
	{time.Hour, "h"},
	{time.Minute, "m"},
}

func (ma maxAgePolicy) String() string {
	d := time.Duration(ma)
	for _, u := range units {
		if d%u.d == 0 {
			return fmt.Sprintf("age() > %d%s", d/u.d, u.suffix)
		}
	}
	return fmt.Sprintf("age() > %d", d/time.Microsecond)
}

func (ma maxAgePolicy) proto() *bttdpb.GcRule {
	// This doesn't handle overflows, etc.
	// Fix this if people care about GC policies over 290 years.
	ns := time.Duration(ma).Nanoseconds()
	return &bttdpb.GcRule{
		Rule: &bttdpb.GcRule_MaxAge{&durpb.Duration{
			Seconds: ns / 1e9,
			Nanos:   int32(ns % 1e9),
		}},
	}
}
