// +build !windows

package idtools

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/unix"
)

type node struct {
	uid int
	gid int
}

func TestMkdirAllAs(t *testing.T) {
	dirName, err := ioutil.TempDir("", "mkdirall")
	if err != nil {
		t.Fatalf("Couldn't create temp dir: %v", err)
	}
	defer os.RemoveAll(dirName)

	testTree := map[string]node{
		"usr":              {0, 0},
		"usr/bin":          {0, 0},
		"lib":              {33, 33},
		"lib/x86_64":       {45, 45},
		"lib/x86_64/share": {1, 1},
	}

	if err := buildTree(dirName, testTree); err != nil {
		t.Fatal(err)
	}

	// test adding a directory to a pre-existing dir; only the new dir is owned by the uid/gid
	if err := MkdirAllAs(filepath.Join(dirName, "usr", "share"), 0755, 99, 99); err != nil {
		t.Fatal(err)
	}
	testTree["usr/share"] = node{99, 99}
	verifyTree, err := readTree(dirName, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := compareTrees(testTree, verifyTree); err != nil {
		t.Fatal(err)
	}

	// test 2-deep new directories--both should be owned by the uid/gid pair
	if err := MkdirAllAs(filepath.Join(dirName, "lib", "some", "other"), 0755, 101, 101); err != nil {
		t.Fatal(err)
	}
	testTree["lib/some"] = node{101, 101}
	testTree["lib/some/other"] = node{101, 101}
	verifyTree, err = readTree(dirName, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := compareTrees(testTree, verifyTree); err != nil {
		t.Fatal(err)
	}

	// test a directory that already exists; should be chowned, but nothing else
	if err := MkdirAllAs(filepath.Join(dirName, "usr"), 0755, 102, 102); err != nil {
		t.Fatal(err)
	}
	testTree["usr"] = node{102, 102}
	verifyTree, err = readTree(dirName, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := compareTrees(testTree, verifyTree); err != nil {
		t.Fatal(err)
	}
}

func TestMkdirAllAndChownNew(t *testing.T) {
	dirName, err := ioutil.TempDir("", "mkdirnew")
	require.NoError(t, err)
	defer os.RemoveAll(dirName)

	testTree := map[string]node{
		"usr":              {0, 0},
		"usr/bin":          {0, 0},
		"lib":              {33, 33},
		"lib/x86_64":       {45, 45},
		"lib/x86_64/share": {1, 1},
	}
	require.NoError(t, buildTree(dirName, testTree))

	// test adding a directory to a pre-existing dir; only the new dir is owned by the uid/gid
	err = MkdirAllAndChownNew(filepath.Join(dirName, "usr", "share"), 0755, IDPair{99, 99})
	require.NoError(t, err)

	testTree["usr/share"] = node{99, 99}
	verifyTree, err := readTree(dirName, "")
	require.NoError(t, err)
	require.NoError(t, compareTrees(testTree, verifyTree))

	// test 2-deep new directories--both should be owned by the uid/gid pair
	err = MkdirAllAndChownNew(filepath.Join(dirName, "lib", "some", "other"), 0755, IDPair{101, 101})
	require.NoError(t, err)
	testTree["lib/some"] = node{101, 101}
	testTree["lib/some/other"] = node{101, 101}
	verifyTree, err = readTree(dirName, "")
	require.NoError(t, err)
	require.NoError(t, compareTrees(testTree, verifyTree))

	// test a directory that already exists; should NOT be chowned
	err = MkdirAllAndChownNew(filepath.Join(dirName, "usr"), 0755, IDPair{102, 102})
	require.NoError(t, err)
	verifyTree, err = readTree(dirName, "")
	require.NoError(t, err)
	require.NoError(t, compareTrees(testTree, verifyTree))
}

func TestMkdirAs(t *testing.T) {

	dirName, err := ioutil.TempDir("", "mkdir")
	if err != nil {
		t.Fatalf("Couldn't create temp dir: %v", err)
	}
	defer os.RemoveAll(dirName)

	testTree := map[string]node{
		"usr": {0, 0},
	}
	if err := buildTree(dirName, testTree); err != nil {
		t.Fatal(err)
	}

	// test a directory that already exists; should just chown to the requested uid/gid
	if err := MkdirAs(filepath.Join(dirName, "usr"), 0755, 99, 99); err != nil {
		t.Fatal(err)
	}
	testTree["usr"] = node{99, 99}
	verifyTree, err := readTree(dirName, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := compareTrees(testTree, verifyTree); err != nil {
		t.Fatal(err)
	}

	// create a subdir under a dir which doesn't exist--should fail
	if err := MkdirAs(filepath.Join(dirName, "usr", "bin", "subdir"), 0755, 102, 102); err == nil {
		t.Fatalf("Trying to create a directory with Mkdir where the parent doesn't exist should have failed")
	}

	// create a subdir under an existing dir; should only change the ownership of the new subdir
	if err := MkdirAs(filepath.Join(dirName, "usr", "bin"), 0755, 102, 102); err != nil {
		t.Fatal(err)
	}
	testTree["usr/bin"] = node{102, 102}
	verifyTree, err = readTree(dirName, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := compareTrees(testTree, verifyTree); err != nil {
		t.Fatal(err)
	}
}

func buildTree(base string, tree map[string]node) error {
	for path, node := range tree {
		fullPath := filepath.Join(base, path)
		if err := os.MkdirAll(fullPath, 0755); err != nil {
			return fmt.Errorf("Couldn't create path: %s; error: %v", fullPath, err)
		}
		if err := os.Chown(fullPath, node.uid, node.gid); err != nil {
			return fmt.Errorf("Couldn't chown path: %s; error: %v", fullPath, err)
		}
	}
	return nil
}

func readTree(base, root string) (map[string]node, error) {
	tree := make(map[string]node)

	dirInfos, err := ioutil.ReadDir(base)
	if err != nil {
		return nil, fmt.Errorf("Couldn't read directory entries for %q: %v", base, err)
	}

	for _, info := range dirInfos {
		s := &unix.Stat_t{}
		if err := unix.Stat(filepath.Join(base, info.Name()), s); err != nil {
			return nil, fmt.Errorf("Can't stat file %q: %v", filepath.Join(base, info.Name()), err)
		}
		tree[filepath.Join(root, info.Name())] = node{int(s.Uid), int(s.Gid)}
		if info.IsDir() {
			// read the subdirectory
			subtree, err := readTree(filepath.Join(base, info.Name()), filepath.Join(root, info.Name()))
			if err != nil {
				return nil, err
			}
			for path, nodeinfo := range subtree {
				tree[path] = nodeinfo
			}
		}
	}
	return tree, nil
}

func compareTrees(left, right map[string]node) error {
	if len(left) != len(right) {
		return fmt.Errorf("Trees aren't the same size")
	}
	for path, nodeLeft := range left {
		if nodeRight, ok := right[path]; ok {
			if nodeRight.uid != nodeLeft.uid || nodeRight.gid != nodeLeft.gid {
				// mismatch
				return fmt.Errorf("mismatched ownership for %q: expected: %d:%d, got: %d:%d", path,
					nodeLeft.uid, nodeLeft.gid, nodeRight.uid, nodeRight.gid)
			}
			continue
		}
		return fmt.Errorf("right tree didn't contain path %q", path)
	}
	return nil
}

func TestParseSubidFileWithNewlinesAndComments(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "parsesubid")
	if err != nil {
		t.Fatal(err)
	}
	fnamePath := filepath.Join(tmpDir, "testsubuid")
	fcontent := `tss:100000:65536
# empty default subuid/subgid file

dockremap:231072:65536`
	if err := ioutil.WriteFile(fnamePath, []byte(fcontent), 0644); err != nil {
		t.Fatal(err)
	}
	ranges, err := parseSubidFile(fnamePath, "dockremap")
	if err != nil {
		t.Fatal(err)
	}
	if len(ranges) != 1 {
		t.Fatalf("wanted 1 element in ranges, got %d instead", len(ranges))
	}
	if ranges[0].Start != 231072 {
		t.Fatalf("wanted 231072, got %d instead", ranges[0].Start)
	}
	if ranges[0].Length != 65536 {
		t.Fatalf("wanted 65536, got %d instead", ranges[0].Length)
	}
}
