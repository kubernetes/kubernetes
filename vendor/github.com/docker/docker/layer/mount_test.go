package layer

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"

	"github.com/docker/docker/pkg/archive"
)

func TestMountInit(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}
	ls, _, cleanup := newTestStore(t)
	defer cleanup()

	basefile := newTestFile("testfile.txt", []byte("base data!"), 0644)
	initfile := newTestFile("testfile.txt", []byte("init data!"), 0777)

	li := initWithFiles(basefile)
	layer, err := createLayer(ls, "", li)
	if err != nil {
		t.Fatal(err)
	}

	mountInit := func(root string) error {
		return initfile.ApplyFile(root)
	}

	rwLayerOpts := &CreateRWLayerOpts{
		InitFunc: mountInit,
	}
	m, err := ls.CreateRWLayer("fun-mount", layer.ChainID(), rwLayerOpts)
	if err != nil {
		t.Fatal(err)
	}

	path, err := m.Mount("")
	if err != nil {
		t.Fatal(err)
	}

	f, err := os.Open(filepath.Join(path, "testfile.txt"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}

	b, err := ioutil.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}

	if expected := "init data!"; string(b) != expected {
		t.Fatalf("Unexpected test file contents %q, expected %q", string(b), expected)
	}

	if fi.Mode().Perm() != 0777 {
		t.Fatalf("Unexpected filemode %o, expecting %o", fi.Mode().Perm(), 0777)
	}
}

func TestMountSize(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}
	ls, _, cleanup := newTestStore(t)
	defer cleanup()

	content1 := []byte("Base contents")
	content2 := []byte("Mutable contents")
	contentInit := []byte("why am I excluded from the size ☹")

	li := initWithFiles(newTestFile("file1", content1, 0644))
	layer, err := createLayer(ls, "", li)
	if err != nil {
		t.Fatal(err)
	}

	mountInit := func(root string) error {
		return newTestFile("file-init", contentInit, 0777).ApplyFile(root)
	}
	rwLayerOpts := &CreateRWLayerOpts{
		InitFunc: mountInit,
	}

	m, err := ls.CreateRWLayer("mount-size", layer.ChainID(), rwLayerOpts)
	if err != nil {
		t.Fatal(err)
	}

	path, err := m.Mount("")
	if err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile(filepath.Join(path, "file2"), content2, 0755); err != nil {
		t.Fatal(err)
	}

	mountSize, err := m.Size()
	if err != nil {
		t.Fatal(err)
	}

	if expected := len(content2); int(mountSize) != expected {
		t.Fatalf("Unexpected mount size %d, expected %d", int(mountSize), expected)
	}
}

func TestMountChanges(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}
	ls, _, cleanup := newTestStore(t)
	defer cleanup()

	basefiles := []FileApplier{
		newTestFile("testfile1.txt", []byte("base data!"), 0644),
		newTestFile("testfile2.txt", []byte("base data!"), 0644),
		newTestFile("testfile3.txt", []byte("base data!"), 0644),
	}
	initfile := newTestFile("testfile1.txt", []byte("init data!"), 0777)

	li := initWithFiles(basefiles...)
	layer, err := createLayer(ls, "", li)
	if err != nil {
		t.Fatal(err)
	}

	mountInit := func(root string) error {
		return initfile.ApplyFile(root)
	}
	rwLayerOpts := &CreateRWLayerOpts{
		InitFunc: mountInit,
	}

	m, err := ls.CreateRWLayer("mount-changes", layer.ChainID(), rwLayerOpts)
	if err != nil {
		t.Fatal(err)
	}

	path, err := m.Mount("")
	if err != nil {
		t.Fatal(err)
	}

	if err := os.Chmod(filepath.Join(path, "testfile1.txt"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile(filepath.Join(path, "testfile1.txt"), []byte("mount data!"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := os.Remove(filepath.Join(path, "testfile2.txt")); err != nil {
		t.Fatal(err)
	}

	if err := os.Chmod(filepath.Join(path, "testfile3.txt"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile(filepath.Join(path, "testfile4.txt"), []byte("mount data!"), 0644); err != nil {
		t.Fatal(err)
	}

	changes, err := m.Changes()
	if err != nil {
		t.Fatal(err)
	}

	if expected := 4; len(changes) != expected {
		t.Fatalf("Wrong number of changes %d, expected %d", len(changes), expected)
	}

	sortChanges(changes)

	assertChange(t, changes[0], archive.Change{
		Path: "/testfile1.txt",
		Kind: archive.ChangeModify,
	})
	assertChange(t, changes[1], archive.Change{
		Path: "/testfile2.txt",
		Kind: archive.ChangeDelete,
	})
	assertChange(t, changes[2], archive.Change{
		Path: "/testfile3.txt",
		Kind: archive.ChangeModify,
	})
	assertChange(t, changes[3], archive.Change{
		Path: "/testfile4.txt",
		Kind: archive.ChangeAdd,
	})
}

func assertChange(t *testing.T, actual, expected archive.Change) {
	if actual.Path != expected.Path {
		t.Fatalf("Unexpected change path %s, expected %s", actual.Path, expected.Path)
	}
	if actual.Kind != expected.Kind {
		t.Fatalf("Unexpected change type %s, expected %s", actual.Kind, expected.Kind)
	}
}

func sortChanges(changes []archive.Change) {
	cs := &changeSorter{
		changes: changes,
	}
	sort.Sort(cs)
}

type changeSorter struct {
	changes []archive.Change
}

func (cs *changeSorter) Len() int {
	return len(cs.changes)
}

func (cs *changeSorter) Swap(i, j int) {
	cs.changes[i], cs.changes[j] = cs.changes[j], cs.changes[i]
}

func (cs *changeSorter) Less(i, j int) bool {
	return cs.changes[i].Path < cs.changes[j].Path
}
