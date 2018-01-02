package store

import (
	"errors"
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"github.com/docker/docker/volume/drivers"
	volumetestutils "github.com/docker/docker/volume/testutils"
)

func TestCreate(t *testing.T) {
	volumedrivers.Register(volumetestutils.NewFakeDriver("fake"), "fake")
	defer volumedrivers.Unregister("fake")
	dir, err := ioutil.TempDir("", "test-create")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	s, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}
	v, err := s.Create("fake1", "fake", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if v.Name() != "fake1" {
		t.Fatalf("Expected fake1 volume, got %v", v)
	}
	if l, _, _ := s.List(); len(l) != 1 {
		t.Fatalf("Expected 1 volume in the store, got %v: %v", len(l), l)
	}

	if _, err := s.Create("none", "none", nil, nil); err == nil {
		t.Fatalf("Expected unknown driver error, got nil")
	}

	_, err = s.Create("fakeerror", "fake", map[string]string{"error": "create error"}, nil)
	expected := &OpErr{Op: "create", Name: "fakeerror", Err: errors.New("create error")}
	if err != nil && err.Error() != expected.Error() {
		t.Fatalf("Expected create fakeError: create error, got %v", err)
	}
}

func TestRemove(t *testing.T) {
	volumedrivers.Register(volumetestutils.NewFakeDriver("fake"), "fake")
	volumedrivers.Register(volumetestutils.NewFakeDriver("noop"), "noop")
	defer volumedrivers.Unregister("fake")
	defer volumedrivers.Unregister("noop")
	dir, err := ioutil.TempDir("", "test-remove")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	s, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}

	// doing string compare here since this error comes directly from the driver
	expected := "no such volume"
	if err := s.Remove(volumetestutils.NoopVolume{}); err == nil || !strings.Contains(err.Error(), expected) {
		t.Fatalf("Expected error %q, got %v", expected, err)
	}

	v, err := s.CreateWithRef("fake1", "fake", "fake", nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := s.Remove(v); !IsInUse(err) {
		t.Fatalf("Expected ErrVolumeInUse error, got %v", err)
	}
	s.Dereference(v, "fake")
	if err := s.Remove(v); err != nil {
		t.Fatal(err)
	}
	if l, _, _ := s.List(); len(l) != 0 {
		t.Fatalf("Expected 0 volumes in the store, got %v, %v", len(l), l)
	}
}

func TestList(t *testing.T) {
	volumedrivers.Register(volumetestutils.NewFakeDriver("fake"), "fake")
	volumedrivers.Register(volumetestutils.NewFakeDriver("fake2"), "fake2")
	defer volumedrivers.Unregister("fake")
	defer volumedrivers.Unregister("fake2")
	dir, err := ioutil.TempDir("", "test-list")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	s, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Create("test", "fake", nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Create("test2", "fake2", nil, nil); err != nil {
		t.Fatal(err)
	}

	ls, _, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(ls) != 2 {
		t.Fatalf("expected 2 volumes, got: %d", len(ls))
	}
	if err := s.Shutdown(); err != nil {
		t.Fatal(err)
	}

	// and again with a new store
	s, err = New(dir)
	if err != nil {
		t.Fatal(err)
	}
	ls, _, err = s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(ls) != 2 {
		t.Fatalf("expected 2 volumes, got: %d", len(ls))
	}
}

func TestFilterByDriver(t *testing.T) {
	volumedrivers.Register(volumetestutils.NewFakeDriver("fake"), "fake")
	volumedrivers.Register(volumetestutils.NewFakeDriver("noop"), "noop")
	defer volumedrivers.Unregister("fake")
	defer volumedrivers.Unregister("noop")
	dir, err := ioutil.TempDir("", "test-filter-driver")
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := s.Create("fake1", "fake", nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Create("fake2", "fake", nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Create("fake3", "noop", nil, nil); err != nil {
		t.Fatal(err)
	}

	if l, _ := s.FilterByDriver("fake"); len(l) != 2 {
		t.Fatalf("Expected 2 volumes, got %v, %v", len(l), l)
	}

	if l, _ := s.FilterByDriver("noop"); len(l) != 1 {
		t.Fatalf("Expected 1 volume, got %v, %v", len(l), l)
	}
}

func TestFilterByUsed(t *testing.T) {
	volumedrivers.Register(volumetestutils.NewFakeDriver("fake"), "fake")
	volumedrivers.Register(volumetestutils.NewFakeDriver("noop"), "noop")
	dir, err := ioutil.TempDir("", "test-filter-used")
	if err != nil {
		t.Fatal(err)
	}

	s, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := s.CreateWithRef("fake1", "fake", "volReference", nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Create("fake2", "fake", nil, nil); err != nil {
		t.Fatal(err)
	}

	vols, _, err := s.List()
	if err != nil {
		t.Fatal(err)
	}

	dangling := s.FilterByUsed(vols, false)
	if len(dangling) != 1 {
		t.Fatalf("expected 1 dangling volume, got %v", len(dangling))
	}
	if dangling[0].Name() != "fake2" {
		t.Fatalf("expected dangling volume fake2, got %s", dangling[0].Name())
	}

	used := s.FilterByUsed(vols, true)
	if len(used) != 1 {
		t.Fatalf("expected 1 used volume, got %v", len(used))
	}
	if used[0].Name() != "fake1" {
		t.Fatalf("expected used volume fake1, got %s", used[0].Name())
	}
}

func TestDerefMultipleOfSameRef(t *testing.T) {
	volumedrivers.Register(volumetestutils.NewFakeDriver("fake"), "fake")
	dir, err := ioutil.TempDir("", "test-same-deref")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	s, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}

	v, err := s.CreateWithRef("fake1", "fake", "volReference", nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := s.GetWithRef("fake1", "fake", "volReference"); err != nil {
		t.Fatal(err)
	}

	s.Dereference(v, "volReference")
	if err := s.Remove(v); err != nil {
		t.Fatal(err)
	}
}
