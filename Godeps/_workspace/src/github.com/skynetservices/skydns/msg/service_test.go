// Copyright (c) 2015 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package msg

import "testing"

func TestSplit255(t *testing.T) {
	xs := split255("abc")
	if len(xs) != 1 && xs[0] != "abc" {
		t.Logf("Failure to split abc")
		t.Fail()
	}
	s := ""
	for i := 0; i < 255; i++ {
		s += "a"
	}
	xs = split255(s)
	if len(xs) != 1 && xs[0] != s {
		t.Logf("failure to split 255 char long string")
		t.Logf("%s %v\n", s, xs)
		t.Fail()
	}
	s += "b"
	xs = split255(s)
	if len(xs) != 2 || xs[1] != "b" {
		t.Logf("failure to split 256 char long string: %d", len(xs))
		t.Logf("%s %v\n", s, xs)
		t.Fail()
	}
	for i := 0; i < 255; i++ {
		s += "a"
	}
	xs = split255(s)
	if len(xs) != 3 || xs[2] != "a" {
		t.Logf("failure to split 510 char long string: %d", len(xs))
		t.Logf("%s %v\n", s, xs)
		t.Fail()
	}
}

func TestGroup(t *testing.T) {
	// Key are in the wrong order, but for this test it does not matter.

	sx := Group(
		[]Service{
			{Host: "127.0.0.1", Group: "g1", Key: "b/sub/dom1/skydns/test"},
			{Host: "127.0.0.2", Group: "g2", Key: "a/dom1/skydns/test"},
		},
	)
	// Expecting to return the shortest key with a Group attribute.
	if len(sx) != 1 {
		t.Fatalf("failure to group zeroth set: %v", sx)
	}
	if sx[0].Key != "a/dom1/skydns/test" {
		t.Fatalf("failure to group zeroth set: %v, wrong Key", sx)
	}

	// Groups disagree, so we will not do anything.
	sx = Group(
		[]Service{
			{Host: "server1", Group: "g1", Key: "region1/skydns/test"},
			{Host: "server2", Group: "g2", Key: "region1/skydns/test"},
		},
	)
	if len(sx) != 2 {
		t.Fatalf("failure to group first set: %v", sx)
	}

	// Group is g1, include only the top-level one.
	sx = Group(
		[]Service{
			{Host: "server1", Group: "g1", Key: "a/dom/region1/skydns/test"},
			{Host: "server2", Group: "g2", Key: "a/subdom/dom/region1/skydns/test"},
		},
	)
	if len(sx) != 1 {
		t.Fatalf("failure to group second set: %v", sx)
	}

	// Groupless services must be included.
	sx = Group(
		[]Service{
			{Host: "server1", Group: "g1", Key: "a/dom/region1/skydns/test"},
			{Host: "server2", Group: "g2", Key: "a/subdom/dom/region1/skydns/test"},
			{Host: "server2", Group: "", Key: "b/subdom/dom/region1/skydns/test"},
		},
	)
	if len(sx) != 2 {
		t.Fatalf("failure to group third set: %v", sx)
	}

	// Empty group on the highest level: include that one also.
	sx = Group(
		[]Service{
			{Host: "server1", Group: "g1", Key: "a/dom/region1/skydns/test"},
			{Host: "server1", Group: "", Key: "b/dom/region1/skydns/test"},
			{Host: "server2", Group: "g2", Key: "a/subdom/dom/region1/skydns/test"},
		},
	)
	if len(sx) != 2 {
		t.Fatalf("failure to group fourth set: %v", sx)
	}

	// Empty group on the highest level: include that one also, and the rest.
	sx = Group(
		[]Service{
			{Host: "server1", Group: "g5", Key: "a/dom/region1/skydns/test"},
			{Host: "server1", Group: "", Key: "b/dom/region1/skydns/test"},
			{Host: "server2", Group: "g5", Key: "a/subdom/dom/region1/skydns/test"},
		},
	)
	if len(sx) != 3 {
		t.Fatalf("failure to group fith set: %v", sx)
	}

	// One group.
	sx = Group(
		[]Service{
			{Host: "server1", Group: "g6", Key: "a/dom/region1/skydns/test"},
		},
	)
	if len(sx) != 1 {
		t.Fatalf("failure to group sixth set: %v", sx)
	}

	// No group, once service
	sx = Group(
		[]Service{
			{Host: "server1", Key: "a/dom/region1/skydns/test"},
		},
	)
	if len(sx) != 1 {
		t.Fatalf("failure to group seventh set: %v", sx)
	}
}
