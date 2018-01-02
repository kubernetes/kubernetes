package main

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func getRepoTopDir(t *testing.T) string {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	wd = filepath.Clean(wd)
	suffix := "hack/integration-cli-on-swarm/host"
	if !strings.HasSuffix(wd, suffix) {
		t.Skipf("cwd seems strange (needs to have suffix %s): %v", suffix, wd)
	}
	return filepath.Clean(filepath.Join(wd, "../../.."))
}

func TestEnumerateTests(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	tests, err := enumerateTests(getRepoTopDir(t))
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(tests)
	t.Logf("enumerated %d test filter strings:", len(tests))
	for _, s := range tests {
		t.Logf("- %q", s)
	}
}

func TestEnumerateTestsForBytes(t *testing.T) {
	b := []byte(`package main
import (
	"github.com/go-check/check"
)

func (s *FooSuite) TestA(c *check.C) {
}

func (s *FooSuite) TestAAA(c *check.C) {
}

func (s *BarSuite) TestBar(c *check.C) {
}

func (x *FooSuite) TestC(c *check.C) {
}

func (*FooSuite) TestD(c *check.C) {
}

// should not be counted
func (s *FooSuite) testE(c *check.C) {
}

// counted, although we don't support ungofmt file
  func   (s *FooSuite)    TestF  (c   *check.C){}
`)
	expected := []string{
		"FooSuite.TestA$",
		"FooSuite.TestAAA$",
		"BarSuite.TestBar$",
		"FooSuite.TestC$",
		"FooSuite.TestD$",
		"FooSuite.TestF$",
	}

	actual, err := enumerateTestsForBytes(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("expected %q, got %q", expected, actual)
	}
}
