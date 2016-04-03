// Copyright (c) 2013 Phillip Bond
// Licensed under the MIT License
// see file LICENSE

package systemstat

import (
	"fmt"
	"testing"
)

var (
	msgFail = "%v method fails. Expects %v, returns %v"
)

func TestGetUptime(t *testing.T) {
	s := getUptime("testdata/uptime")
	if s.Uptime != 18667.53 {
		t.Errorf(msgFail, "getUptime", "18667.53", s.Uptime)
	}
}

func TestGetLoadAvgSample(t *testing.T) {
	s := getLoadAvgSample("testdata/loadavg")
	fmt.Printf("%#v\n", s)
	if s.One != 0.1 {
		t.Errorf(msgFail, "getUptime", "0.1", s.One)
	}
	if s.Five != 0.15 {
		t.Errorf(msgFail, "getUptime", "0.15", s.Five)
	}
	if s.Fifteen != 0.14 {
		t.Errorf(msgFail, "getUptime", "0.14", s.Fifteen)
	}
}
