package filesystem

import (
	"io"
	"os"
	"path"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"
	"github.com/go-git/go-git/v5/utils/merkletrie/noder"

	"github.com/go-git/go-billy/v5"
)

var ignore = map[string]bool{
	".git": true,
}

// The node represents a file or a directory in a billy.Filesystem. It
// implements the interface noder.Noder of merkletrie package.
//
// This implementation implements a "standard" hash method being able to be
// compared with any other noder.Noder implementation inside of go-git.
type node struct {
	fs         billy.Filesystem
	submodules map[string]plumbing.Hash

	path     string
	hash     []byte
	children []noder.Noder
	isDir    bool
}

// NewRootNode returns the root node based on a given billy.Filesystem.
//
// In order to provide the submodule hash status, a map[string]plumbing.Hash
// should be provided where the key is the path of the submodule and the commit
// of the submodule HEAD
func NewRootNode(
	fs billy.Filesystem,
	submodules map[string]plumbing.Hash,
) noder.Noder {
	return &node{fs: fs, submodules: submodules, isDir: true}
}

// Hash the hash of a filesystem is the result of concatenating the computed
// plumbing.Hash of the file as a Blob and its plumbing.FileMode; that way the
// difftree algorithm will detect changes in the contents of files and also in
// their mode.
//
// The hash of a directory is always a 24-bytes slice of zero values
func (n *node) Hash() []byte {
	return n.hash
}

func (n *node) Name() string {
	return path.Base(n.path)
}

func (n *node) IsDir() bool {
	return n.isDir
}

func (n *node) Children() ([]noder.Noder, error) {
	if err := n.calculateChildren(); err != nil {
		return nil, err
	}

	return n.children, nil
}

func (n *node) NumChildren() (int, error) {
	if err := n.calculateChildren(); err != nil {
		return -1, err
	}

	return len(n.children), nil
}

func (n *node) calculateChildren() error {
	if !n.IsDir() {
		return nil
	}

	if len(n.children) != 0 {
		return nil
	}

	files, err := n.fs.ReadDir(n.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	for _, file := range files {
		if _, ok := ignore[file.Name()]; ok {
			continue
		}

		c, err := n.newChildNode(file)
		if err != nil {
			return err
		}

		n.children = append(n.children, c)
	}

	return nil
}

func (n *node) newChildNode(file os.FileInfo) (*node, error) {
	path := path.Join(n.path, file.Name())

	hash, err := n.calculateHash(path, file)
	if err != nil {
		return nil, err
	}

	node := &node{
		fs:         n.fs,
		submodules: n.submodules,

		path:  path,
		hash:  hash,
		isDir: file.IsDir(),
	}

	if hash, isSubmodule := n.submodules[path]; isSubmodule {
		node.hash = append(hash[:], filemode.Submodule.Bytes()...)
		node.isDir = false
	}

	return node, nil
}

func (n *node) calculateHash(path string, file os.FileInfo) ([]byte, error) {
	if file.IsDir() {
		return make([]byte, 24), nil
	}

	var hash plumbing.Hash
	var err error
	if file.Mode()&os.ModeSymlink != 0 {
		hash, err = n.doCalculateHashForSymlink(path, file)
	} else {
		hash, err = n.doCalculateHashForRegular(path, file)
	}

	if err != nil {
		return nil, err
	}

	mode, err := filemode.NewFromOSFileMode(file.Mode())
	if err != nil {
		return nil, err
	}

	return append(hash[:], mode.Bytes()...), nil
}

func (n *node) doCalculateHashForRegular(path string, file os.FileInfo) (plumbing.Hash, error) {
	f, err := n.fs.Open(path)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	defer f.Close()

	h := plumbing.NewHasher(plumbing.BlobObject, file.Size())
	if _, err := io.Copy(h, f); err != nil {
		return plumbing.ZeroHash, err
	}

	return h.Sum(), nil
}

func (n *node) doCalculateHashForSymlink(path string, file os.FileInfo) (plumbing.Hash, error) {
	target, err := n.fs.Readlink(path)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	h := plumbing.NewHasher(plumbing.BlobObject, file.Size())
	if _, err := h.Write([]byte(target)); err != nil {
		return plumbing.ZeroHash, err
	}

	return h.Sum(), nil
}

func (n *node) String() string {
	return n.path
}
