// Copyright (C) 2012 Numerotron Inc.
// Use of this source code is governed by an MIT-style license
// that can be found in the LICENSE file.

package stathat

import (
	"testing"
)

func TestNewEZStatCount(t *testing.T) {
	setTesting()
	x := newEZStatCount("abc", "pc@pc.com", 1)
	if x == nil {
		t.Fatalf("expected a StatReport object")
	}
	if x.statType != kcounter {
		t.Errorf("expected counter")
	}
	if x.apiType != ez {
		t.Errorf("expected EZ api")
	}
	if x.StatKey != "abc" {
		t.Errorf("expected abc")
	}
	if x.UserKey != "pc@pc.com" {
		t.Errorf("expected pc@pc.com")
	}
	if x.Value != 1.0 {
		t.Errorf("expected 1.0")
	}
	if x.Timestamp != 0 {
		t.Errorf("expected 0")
	}
}

func TestNewEZStatValue(t *testing.T) {
	setTesting()
	x := newEZStatValue("abc", "pc@pc.com", 3.14159)
	if x == nil {
		t.Fatalf("expected a StatReport object")
	}
	if x.statType != kvalue {
		t.Errorf("expected value")
	}
	if x.apiType != ez {
		t.Errorf("expected EZ api")
	}
	if x.StatKey != "abc" {
		t.Errorf("expected abc")
	}
	if x.UserKey != "pc@pc.com" {
		t.Errorf("expected pc@pc.com")
	}
	if x.Value != 3.14159 {
		t.Errorf("expected 3.14159")
	}
}

func TestNewClassicStatCount(t *testing.T) {
	setTesting()
	x := newClassicStatCount("statkey", "userkey", 1)
	if x == nil {
		t.Fatalf("expected a StatReport object")
	}
	if x.statType != kcounter {
		t.Errorf("expected counter")
	}
	if x.apiType != classic {
		t.Errorf("expected CLASSIC api")
	}
	if x.StatKey != "statkey" {
		t.Errorf("expected statkey")
	}
	if x.UserKey != "userkey" {
		t.Errorf("expected userkey")
	}
	if x.Value != 1.0 {
		t.Errorf("expected 1.0")
	}
	if x.Timestamp != 0 {
		t.Errorf("expected 0")
	}
}

func TestNewClassicStatValue(t *testing.T) {
	setTesting()
	x := newClassicStatValue("statkey", "userkey", 2.28)
	if x == nil {
		t.Fatalf("expected a StatReport object")
	}
	if x.statType != kvalue {
		t.Errorf("expected value")
	}
	if x.apiType != classic {
		t.Errorf("expected CLASSIC api")
	}
	if x.StatKey != "statkey" {
		t.Errorf("expected statkey")
	}
	if x.UserKey != "userkey" {
		t.Errorf("expected userkey")
	}
	if x.Value != 2.28 {
		t.Errorf("expected 2.28")
	}
}

func TestURLValues(t *testing.T) {
	setTesting()
	x := newEZStatCount("abc", "pc@pc.com", 1)
	v := x.values()
	if v == nil {
		t.Fatalf("expected url values")
	}
	if v.Get("stat") != "abc" {
		t.Errorf("expected abc")
	}
	if v.Get("ezkey") != "pc@pc.com" {
		t.Errorf("expected pc@pc.com")
	}
	if v.Get("count") != "1" {
		t.Errorf("expected count of 1")
	}

	y := newEZStatValue("abc", "pc@pc.com", 3.14159)
	v = y.values()
	if v == nil {
		t.Fatalf("expected url values")
	}
	if v.Get("stat") != "abc" {
		t.Errorf("expected abc")
	}
	if v.Get("ezkey") != "pc@pc.com" {
		t.Errorf("expected pc@pc.com")
	}
	if v.Get("value") != "3.14159" {
		t.Errorf("expected value of 3.14159")
	}

	a := newClassicStatCount("statkey", "userkey", 1)
	v = a.values()
	if v == nil {
		t.Fatalf("expected url values")
	}
	if v.Get("key") != "statkey" {
		t.Errorf("expected statkey")
	}
	if v.Get("ukey") != "userkey" {
		t.Errorf("expected userkey")
	}
	if v.Get("count") != "1" {
		t.Errorf("expected count of 1")
	}

	b := newClassicStatValue("statkey", "userkey", 2.28)
	v = b.values()
	if v == nil {
		t.Fatalf("expected url values")
	}
	if v.Get("key") != "statkey" {
		t.Errorf("expected statkey")
	}
	if v.Get("ukey") != "userkey" {
		t.Errorf("expected userkey")
	}
	if v.Get("value") != "2.28" {
		t.Errorf("expected value of 2.28")
	}
}

func TestPaths(t *testing.T) {
	if ez.path(kcounter) != "/ez" {
		t.Errorf("expected /ez")
	}
	if ez.path(kvalue) != "/ez" {
		t.Errorf("expected /ez")
	}
	if classic.path(kcounter) != "/c" {
		t.Errorf("expected /c")
	}
	if classic.path(kvalue) != "/v" {
		t.Errorf("expected /v")
	}

	x := newEZStatCount("abc", "pc@pc.com", 1)
	if x.path() != "/ez" {
		t.Errorf("expected /ez")
	}
	y := newEZStatValue("abc", "pc@pc.com", 3.14159)
	if y.path() != "/ez" {
		t.Errorf("expected /ez")
	}
	a := newClassicStatCount("statkey", "userkey", 1)
	if a.path() != "/c" {
		t.Errorf("expected /c")
	}
	b := newClassicStatValue("statkey", "userkey", 2.28)
	if b.path() != "/v" {
		t.Errorf("expected /v")
	}
}

func TestPosts(t *testing.T) {
	setTesting()
	Verbose = true
	PostCountOne("statkey", "userkey")
	p := <-testPostChannel
	if p.url != "http://api.stathat.com/c" {
		t.Errorf("expected classic count url")
	}
	if p.values.Get("key") != "statkey" {
		t.Errorf("expected statkey")
	}
	if p.values.Get("ukey") != "userkey" {
		t.Errorf("expected userkey")
	}
	if p.values.Get("count") != "1" {
		t.Errorf("expected count of 1")
	}

	PostCount("statkey", "userkey", 13)
	p = <-testPostChannel
	if p.url != "http://api.stathat.com/c" {
		t.Errorf("expected classic count url")
	}
	if p.values.Get("key") != "statkey" {
		t.Errorf("expected statkey")
	}
	if p.values.Get("ukey") != "userkey" {
		t.Errorf("expected userkey")
	}
	if p.values.Get("count") != "13" {
		t.Errorf("expected count of 13")
	}

	PostValue("statkey", "userkey", 9.312)
	p = <-testPostChannel
	if p.url != "http://api.stathat.com/v" {
		t.Errorf("expected classic value url")
	}
	if p.values.Get("key") != "statkey" {
		t.Errorf("expected statkey")
	}
	if p.values.Get("ukey") != "userkey" {
		t.Errorf("expected userkey")
	}
	if p.values.Get("value") != "9.312" {
		t.Errorf("expected value of 9.312")
	}

	PostEZCountOne("a stat", "pc@pc.com")
	p = <-testPostChannel
	if p.url != "http://api.stathat.com/ez" {
		t.Errorf("expected ez url")
	}
	if p.values.Get("stat") != "a stat" {
		t.Errorf("expected a stat")
	}
	if p.values.Get("ezkey") != "pc@pc.com" {
		t.Errorf("expected pc@pc.com")
	}
	if p.values.Get("count") != "1" {
		t.Errorf("expected count of 1")
	}

	PostEZCount("a stat", "pc@pc.com", 213)
	p = <-testPostChannel
	if p.url != "http://api.stathat.com/ez" {
		t.Errorf("expected ez url")
	}
	if p.values.Get("stat") != "a stat" {
		t.Errorf("expected a stat")
	}
	if p.values.Get("ezkey") != "pc@pc.com" {
		t.Errorf("expected pc@pc.com")
	}
	if p.values.Get("count") != "213" {
		t.Errorf("expected count of 213")
	}

	PostEZValue("a stat", "pc@pc.com", 2.13)
	p = <-testPostChannel
	if p.url != "http://api.stathat.com/ez" {
		t.Errorf("expected ez url")
	}
	if p.values.Get("stat") != "a stat" {
		t.Errorf("expected a stat")
	}
	if p.values.Get("ezkey") != "pc@pc.com" {
		t.Errorf("expected pc@pc.com")
	}
	if p.values.Get("value") != "2.13" {
		t.Errorf("expected value of 2.13")
	}

	PostCountTime("statkey", "userkey", 13, 100000)
	p = <-testPostChannel
	if p.values.Get("t") != "100000" {
		t.Errorf("expected t value of 100000, got %s", p.values.Get("t"))
	}

	PostValueTime("statkey", "userkey", 9.312, 200000)
	p = <-testPostChannel
	if p.values.Get("t") != "200000" {
		t.Errorf("expected t value of 200000, got %s", p.values.Get("t"))
	}

	PostEZCountTime("a stat", "pc@pc.com", 213, 300000)
	p = <-testPostChannel
	if p.values.Get("t") != "300000" {
		t.Errorf("expected t value of 300000, got %s", p.values.Get("t"))
	}

	PostEZValueTime("a stat", "pc@pc.com", 2.13, 400000)
	p = <-testPostChannel
	if p.values.Get("t") != "400000" {
		t.Errorf("expected t value of 400000, got %s", p.values.Get("t"))
	}
}
