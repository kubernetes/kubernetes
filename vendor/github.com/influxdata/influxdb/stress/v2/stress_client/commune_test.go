package stressClient

import (
	"testing"
)

func TestCommunePoint(t *testing.T) {
	comm := newCommune(5)
	pt := "write,tag=tagVal fooField=5 1460912595"
	comm.ch <- pt
	point := comm.point("s")
	if point.Name() != "write" {
		t.Errorf("expected: write\ngot: %v", point.Name())
	}
	if point.Tags().GetString("tag") != "tagVal" {
		t.Errorf("expected: tagVal\ngot: %v", point.Tags().GetString("tag"))
	}
	if int(point.Fields()["fooField"].(float64)) != 5 {
		t.Errorf("expected: 5\ngot: %v\n", point.Fields()["fooField"])
	}
	// Make sure commune returns the prev point
	comm.ch <- ""
	point = comm.point("s")
	if point.Name() != "write" {
		t.Errorf("expected: write\ngot: %v", point.Name())
	}
	if point.Tags().GetString("tag") != "tagVal" {
		t.Errorf("expected: tagVal\ngot: %v", point.Tags().GetString("tag"))
	}
	if int(point.Fields()["fooField"].(float64)) != 5 {
		t.Errorf("expected: 5\ngot: %v\n", point.Fields()["fooField"])
	}
}

func TestSetCommune(t *testing.T) {
	sf, _, _ := NewTestStressTest()
	ch := sf.SetCommune("foo_name")
	ch <- "write,tag=tagVal fooField=5 1460912595"
	pt := sf.GetPoint("foo_name", "s")
	if pt.Name() != "write" {
		t.Errorf("expected: write\ngot: %v", pt.Name())
	}
	if pt.Tags().GetString("tag") != "tagVal" {
		t.Errorf("expected: tagVal\ngot: %v", pt.Tags().GetString("tag"))
	}
	if int(pt.Fields()["fooField"].(float64)) != 5 {
		t.Errorf("expected: 5\ngot: %v\n", pt.Fields()["fooField"])
	}
}
