// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/errors"
)

var _ File = &fsNode{}
var _ FileSystem = &fsNode{}

// fsNode is either a file or a directory.
type fsNode struct {
	// What node owns me?
	parent *fsNode

	// Value to return as the Name() when the
	// parent is nil.
	nilParentName string

	// A directory mapping names to nodes.
	// If dir is nil, then self node is a file.
	// If dir is non-nil, then self node is a directory,
	// albeit possibly an empty directory.
	dir map[string]*fsNode

	// if this node is a file, this is the content.
	content []byte

	// if offset is not nil the file is open and it tracks
	// the current file offset.
	offset *int
}

// MakeEmptyDirInMemory returns an empty directory.
// The paths of nodes in this object will never
// report a leading Separator, meaning they
// aren't "absolute" in the sense defined by
// https://golang.org/pkg/path/filepath/#IsAbs.
func MakeEmptyDirInMemory() *fsNode {
	return &fsNode{
		dir: make(map[string]*fsNode),
	}
}

// MakeFsInMemory returns an empty 'file system'.
// The paths of nodes in this object will always
// report a leading Separator, meaning they
// are "absolute" in the sense defined by
// https://golang.org/pkg/path/filepath/#IsAbs.
// This is a relevant difference when using Walk,
// Glob, Match, etc.
func MakeFsInMemory() FileSystem {
	return &fsNode{
		nilParentName: Separator,
		dir:           make(map[string]*fsNode),
	}
}

// Name returns the name of the node.
func (n *fsNode) Name() string {
	if n.parent == nil {
		// Unable to lookup name in parent.
		return n.nilParentName
	}
	if !n.parent.isNodeADir() {
		log.Fatal("parent not a dir")
	}
	for key, value := range n.parent.dir {
		if value == n {
			return key
		}
	}
	log.Fatal("unable to find fsNode name")
	return ""
}

// Path returns the full path to the node.
func (n *fsNode) Path() string {
	if n.parent == nil {
		return n.nilParentName
	}
	if !n.parent.isNodeADir() {
		log.Fatal("parent not a dir, structural error")
	}
	return filepath.Join(n.parent.Path(), n.Name())
}

// mySplit trims trailing separators from the directory
// result of filepath.Split.
func mySplit(s string) (string, string) {
	dName, fName := filepath.Split(s)
	return StripTrailingSeps(dName), fName
}

func (n *fsNode) addFile(name string, c []byte) (result *fsNode, err error) {
	parent := n
	dName, fileName := mySplit(name)
	if dName != "" {
		parent, err = parent.addDir(dName)
		if err != nil {
			return nil, err
		}
	}
	if !isLegalFileNameForCreation(fileName) {
		return nil, fmt.Errorf(
			"illegal name '%s' in file creation", fileName)
	}
	result, ok := parent.dir[fileName]
	if ok {
		// File already exists; overwrite it.
		if result.offset != nil {
			return nil, fmt.Errorf("cannot add already opened file '%s'", n.Path())
		}
		result.content = append(result.content[:0], c...)
		return result, nil
	}
	result = &fsNode{
		content: append([]byte(nil), c...),
		parent:  parent,
	}
	parent.dir[fileName] = result
	return result, nil
}

// Create implements FileSystem.
// Create makes an empty file.
func (n *fsNode) Create(path string) (result File, err error) {
	f, err := n.AddFile(path, nil)
	if err != nil {
		return f, err
	}
	f.offset = new(int)
	return f, nil
}

// WriteFile implements FileSystem.
func (n *fsNode) WriteFile(path string, d []byte) error {
	_, err := n.AddFile(path, d)
	return err
}

// AddFile adds a file and any necessary containing
// directories to the node.
func (n *fsNode) AddFile(
	name string, c []byte) (result *fsNode, err error) {
	if n.dir == nil {
		return nil, fmt.Errorf(
			"cannot add a file to a non-directory '%s'", n.Name())
	}
	return n.addFile(cleanQueryPath(name), c)
}

func (n *fsNode) addDir(path string) (result *fsNode, err error) {
	parent := n
	dName, subDirName := mySplit(path)
	if dName != "" {
		parent, err = n.addDir(dName)
		if err != nil {
			return nil, err
		}
	}
	switch subDirName {
	case "", SelfDir:
		return n, nil
	case ParentDir:
		if n.parent == nil {
			return nil, fmt.Errorf(
				"cannot add a directory above '%s'", n.Path())
		}
		return n.parent, nil
	default:
		if !isLegalFileNameForCreation(subDirName) {
			return nil, fmt.Errorf(
				"illegal name '%s' in directory creation", subDirName)
		}
		result, ok := parent.dir[subDirName]
		if ok {
			if result.isNodeADir() {
				// it's already there.
				return result, nil
			}
			return nil, fmt.Errorf(
				"cannot make dir '%s'; a file of that name already exists in '%s'",
				subDirName, parent.Name())
		}
		result = &fsNode{
			dir:    make(map[string]*fsNode),
			parent: parent,
		}
		parent.dir[subDirName] = result
		return result, nil
	}
}

// Mkdir implements FileSystem.
// Mkdir creates a directory.
func (n *fsNode) Mkdir(path string) error {
	_, err := n.AddDir(path)
	return err
}

// MkdirAll implements FileSystem.
// MkdirAll creates a directory.
func (n *fsNode) MkdirAll(path string) error {
	_, err := n.AddDir(path)
	return err
}

// AddDir adds a directory to the node, not complaining
// if it is already there.
func (n *fsNode) AddDir(path string) (result *fsNode, err error) {
	if n.dir == nil {
		return nil, fmt.Errorf(
			"cannot add a directory to file node '%s'", n.Name())
	}
	return n.addDir(cleanQueryPath(path))
}

// CleanedAbs implements FileSystem.
func (n *fsNode) CleanedAbs(path string) (ConfirmedDir, string, error) {
	node, err := n.Find(path)
	if err != nil {
		return "", "", errors.WrapPrefixf(err, "unable to clean")
	}
	if node == nil {
		return "", "", notExistError(path)
	}
	if node.isNodeADir() {
		return ConfirmedDir(node.Path()), "", nil
	}
	return ConfirmedDir(node.parent.Path()), node.Name(), nil
}

// Exists implements FileSystem.
// Exists returns true if the path exists.
func (n *fsNode) Exists(path string) bool {
	if !n.isNodeADir() {
		return n.Name() == path
	}
	result, err := n.Find(path)
	if err != nil {
		return false
	}
	return result != nil
}

func cleanQueryPath(path string) string {
	// Always ignore leading separator?
	// Remember that filepath.Clean returns "." if
	// given an empty string argument.
	return filepath.Clean(StripLeadingSeps(path))
}

// Find finds the given node, else nil if not found.
// Return error on structural/argument errors.
func (n *fsNode) Find(path string) (*fsNode, error) {
	if !n.isNodeADir() {
		return nil, fmt.Errorf("can only find inside a dir")
	}
	if path == "" {
		// Special case; check *before* cleaning and *before*
		// comparison to nilParentName.
		return nil, nil
	}
	if (n.parent == nil && path == n.nilParentName) || path == SelfDir {
		// Special case
		return n, nil
	}
	return n.findIt(cleanQueryPath(path))
}

func (n *fsNode) findIt(path string) (result *fsNode, err error) {
	parent := n
	dName, item := mySplit(path)
	if dName != "" {
		parent, err = n.findIt(dName)
		if err != nil {
			return nil, err
		}
		if parent == nil {
			// all done, target doesn't exist.
			return nil, nil
		}
	}
	if !parent.isNodeADir() {
		return nil, fmt.Errorf("'%s' is not a directory", parent.Path())
	}
	return parent.dir[item], nil
}

// RemoveAll implements FileSystem.
// RemoveAll removes an item and everything it contains.
func (n *fsNode) RemoveAll(path string) error {
	result, err := n.Find(path)
	if err != nil {
		return err
	}
	if result == nil {
		// If the path doesn't exist, no need to remove anything.
		return nil
	}
	return result.Remove()
}

// Remove drop the node, and everything it contains, from its parent.
func (n *fsNode) Remove() error {
	if n.parent == nil {
		return fmt.Errorf("cannot remove a root node")
	}
	if !n.parent.isNodeADir() {
		log.Fatal("parent not a dir")
	}
	for key, value := range n.parent.dir {
		if value == n {
			delete(n.parent.dir, key)
			return nil
		}
	}
	log.Fatal("unable to find self in parent")
	return nil
}

// isNodeADir returns true if the node is a directory.
// Cannot collide with the poorly named "IsDir".
func (n *fsNode) isNodeADir() bool {
	return n.dir != nil
}

// IsDir implements FileSystem.
// IsDir returns true if the argument resolves
// to a directory rooted at the node.
func (n *fsNode) IsDir(path string) bool {
	result, err := n.Find(path)
	if err != nil || result == nil {
		return false
	}
	return result.isNodeADir()
}

// ReadDir implements FileSystem.
func (n *fsNode) ReadDir(path string) ([]string, error) {
	if !n.Exists(path) {
		return nil, notExistError(path)
	}
	if !n.IsDir(path) {
		return nil, fmt.Errorf("%s is not a directory", path)
	}

	dir, err := n.Find(path)
	if err != nil {
		return nil, err
	}
	if dir == nil {
		return nil, fmt.Errorf("could not find directory %s", path)
	}

	keys := make([]string, len(dir.dir))
	i := 0
	for k := range dir.dir {
		keys[i] = k
		i++
	}
	return keys, nil
}

// Size returns the size of the node.
func (n *fsNode) Size() int64 {
	if n.isNodeADir() {
		return int64(len(n.dir))
	}
	return int64(len(n.content))
}

// Open implements FileSystem.
// Open opens the node in read-write mode and sets the offset its start.
// Writing right after opening the file will replace the original content
// and move the offset forward, as with a file opened with O_RDWR | O_CREATE.
//
// As an example, let's consider a file with content "content":
// - open: sets offset to start, content is "content"
// - write "@": offset increases by one, the content is now "@ontent"
// - read the rest: since offset is 1, the read operation returns "ontent"
// - write "$": offset is at EOF, so "$" is appended and content is now "@ontent$"
// - read the rest: returns 0 bytes and EOF
// - close: the content is still "@ontent$"
func (n *fsNode) Open(path string) (File, error) {
	result, err := n.Find(path)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, notExistError(path)
	}
	if result.offset != nil {
		return nil, fmt.Errorf("cannot open previously opened file '%s'", path)
	}
	result.offset = new(int)
	return result, nil
}

// Close marks the node closed.
func (n *fsNode) Close() error {
	if n.offset == nil {
		return fmt.Errorf("cannot close already closed file '%s'", n.Path())
	}
	n.offset = nil
	return nil
}

// ReadFile implements FileSystem.
func (n *fsNode) ReadFile(path string) (c []byte, err error) {
	result, err := n.Find(path)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, notExistError(path)
	}
	if result.isNodeADir() {
		return nil, fmt.Errorf("cannot read content from non-file '%s'", n.Path())
	}
	c = make([]byte, len(result.content))
	copy(c, result.content)
	return c, nil
}

// Read returns the content of the file node.
func (n *fsNode) Read(d []byte) (c int, err error) {
	if n.isNodeADir() {
		return 0, fmt.Errorf(
			"cannot read content from non-file '%s'", n.Path())
	}
	if n.offset == nil {
		return 0, fmt.Errorf("cannot read from closed file '%s'", n.Path())
	}

	rest := n.content[*n.offset:]
	if len(d) < len(rest) {
		rest = rest[:len(d)]
	} else {
		err = io.EOF
	}
	copy(d, rest)
	*n.offset += len(rest)
	return len(rest), err
}

// Write saves the contents of the argument to the file node.
func (n *fsNode) Write(p []byte) (c int, err error) {
	if n.isNodeADir() {
		return 0, fmt.Errorf(
			"cannot write content to non-file '%s'", n.Path())
	}
	if n.offset == nil {
		return 0, fmt.Errorf("cannot write to closed file '%s'", n.Path())
	}
	n.content = append(n.content[:*n.offset], p...)
	*n.offset = len(n.content)
	return len(p), nil
}

// ContentMatches returns true if v matches fake file's content.
func (n *fsNode) ContentMatches(v []byte) bool {
	return bytes.Equal(v, n.content)
}

// GetContent the content of a fake file.
func (n *fsNode) GetContent() []byte {
	return n.content
}

// Stat returns an instance of FileInfo.
func (n *fsNode) Stat() (os.FileInfo, error) {
	return fileInfo{node: n}, nil
}

// Walk implements FileSystem.
func (n *fsNode) Walk(path string, walkFn filepath.WalkFunc) error {
	result, err := n.Find(path)
	if err != nil {
		return err
	}
	if result == nil {
		return notExistError(path)
	}
	return result.WalkMe(walkFn)
}

// Walk runs the given walkFn on each node.
func (n *fsNode) WalkMe(walkFn filepath.WalkFunc) error {
	fi, err := n.Stat()
	// always visit self first
	err = walkFn(n.Path(), fi, err)
	if !n.isNodeADir() {
		// it's a file, so nothing more to do
		return err
	}
	// process self as a directory
	if err == filepath.SkipDir {
		return nil
	}
	// Walk is supposed to visit in lexical order.
	for _, k := range n.sortedDirEntries() {
		if err := n.dir[k].WalkMe(walkFn); err != nil {
			if err == filepath.SkipDir {
				// stop processing this directory
				break
			}
			// bail out completely
			return err
		}
	}
	return nil
}

func (n *fsNode) sortedDirEntries() []string {
	keys := make([]string, len(n.dir))
	i := 0
	for k := range n.dir {
		keys[i] = k
		i++
	}
	sort.Strings(keys)
	return keys
}

// FileCount returns a count of files.
// Directories, empty or otherwise, not counted.
func (n *fsNode) FileCount() int {
	count := 0
	n.WalkMe(func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			count++
		}
		return nil
	})
	return count
}

func (n *fsNode) DebugPrint() {
	n.WalkMe(func(path string, info os.FileInfo, err error) error {
		if err != nil {
			fmt.Printf("err '%v' at path %q\n", err, path)
			return nil
		}
		if info.IsDir() {
			if info.Size() == 0 {
				fmt.Println("empty dir: " + path)
			}
		} else {
			fmt.Println("     file: " + path)
		}
		return nil
	})
}

var legalFileNamePattern = regexp.MustCompile("^[a-zA-Z0-9-_.:]+$")

// This rules enforced here should be simpler and tighter
// than what's allowed on a real OS.
// Should be fine for testing or in-memory purposes.
func isLegalFileNameForCreation(n string) bool {
	if n == "" || n == SelfDir || !legalFileNamePattern.MatchString(n) {
		return false
	}
	return !strings.Contains(n, ParentDir)
}

// RegExpGlob returns a list of file paths matching the regexp.
// Excludes directories.
func (n *fsNode) RegExpGlob(pattern string) ([]string, error) {
	var result []string
	var expression = regexp.MustCompile(pattern)
	err := n.WalkMe(func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			if expression.MatchString(path) {
				result = append(result, path)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(result)
	return result, nil
}

// Glob implements FileSystem.
// Glob returns the list of file paths matching
// per filepath.Match semantics, i.e. unlike RegExpGlob,
// Match("foo/a*") will not match sub-sub directories of foo.
// This is how /bin/ls behaves.
func (n *fsNode) Glob(pattern string) ([]string, error) {
	var result []string
	var allFiles []string
	err := n.WalkMe(func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			match, err := filepath.Match(pattern, path)
			if err != nil {
				return err
			}
			if match {
				allFiles = append(allFiles, path)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if IsHiddenFilePath(pattern) {
		result = allFiles
	} else {
		result = RemoveHiddenFiles(allFiles)
	}
	sort.Strings(result)
	return result, nil
}

// notExistError indicates that a file or directory does not exist.
// Unwrapping returns os.ErrNotExist so errors.Is(err, os.ErrNotExist) works correctly.
type notExistError string

func (err notExistError) Error() string { return fmt.Sprintf("'%s' doesn't exist", string(err)) }
func (err notExistError) Unwrap() error { return os.ErrNotExist }
