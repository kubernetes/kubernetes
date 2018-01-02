package remotecontext

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/docker/docker/pkg/symlink"
	iradix "github.com/hashicorp/go-immutable-radix"
	"github.com/pkg/errors"
	"github.com/tonistiigi/fsutil"
)

type hashed interface {
	Hash() string
}

// CachableSource is a source that contains cache records for its contents
type CachableSource struct {
	mu   sync.Mutex
	root string
	tree *iradix.Tree
	txn  *iradix.Txn
}

// NewCachableSource creates new CachableSource
func NewCachableSource(root string) *CachableSource {
	ts := &CachableSource{
		tree: iradix.New(),
		root: root,
	}
	return ts
}

// MarshalBinary marshals current cache information to a byte array
func (cs *CachableSource) MarshalBinary() ([]byte, error) {
	b := TarsumBackup{Hashes: make(map[string]string)}
	root := cs.getRoot()
	root.Walk(func(k []byte, v interface{}) bool {
		b.Hashes[string(k)] = v.(*fileInfo).sum
		return false
	})
	return b.Marshal()
}

// UnmarshalBinary decodes cache information for presented byte array
func (cs *CachableSource) UnmarshalBinary(data []byte) error {
	var b TarsumBackup
	if err := b.Unmarshal(data); err != nil {
		return err
	}
	txn := iradix.New().Txn()
	for p, v := range b.Hashes {
		txn.Insert([]byte(p), &fileInfo{sum: v})
	}
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.tree = txn.Commit()
	return nil
}

// Scan rescans the cache information from the file system
func (cs *CachableSource) Scan() error {
	lc, err := NewLazySource(cs.root)
	if err != nil {
		return err
	}
	txn := iradix.New().Txn()
	err = filepath.Walk(cs.root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return errors.Wrapf(err, "failed to walk %s", path)
		}
		rel, err := Rel(cs.root, path)
		if err != nil {
			return err
		}
		h, err := lc.Hash(rel)
		if err != nil {
			return err
		}
		txn.Insert([]byte(rel), &fileInfo{sum: h})
		return nil
	})
	if err != nil {
		return err
	}
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.tree = txn.Commit()
	return nil
}

// HandleChange notifies the source about a modification operation
func (cs *CachableSource) HandleChange(kind fsutil.ChangeKind, p string, fi os.FileInfo, err error) (retErr error) {
	cs.mu.Lock()
	if cs.txn == nil {
		cs.txn = cs.tree.Txn()
	}
	if kind == fsutil.ChangeKindDelete {
		cs.txn.Delete([]byte(p))
		cs.mu.Unlock()
		return
	}

	h, ok := fi.(hashed)
	if !ok {
		cs.mu.Unlock()
		return errors.Errorf("invalid fileinfo: %s", p)
	}

	hfi := &fileInfo{
		sum: h.Hash(),
	}
	cs.txn.Insert([]byte(p), hfi)
	cs.mu.Unlock()
	return nil
}

func (cs *CachableSource) getRoot() *iradix.Node {
	cs.mu.Lock()
	if cs.txn != nil {
		cs.tree = cs.txn.Commit()
		cs.txn = nil
	}
	t := cs.tree
	cs.mu.Unlock()
	return t.Root()
}

// Close closes the source
func (cs *CachableSource) Close() error {
	return nil
}

func (cs *CachableSource) normalize(path string) (cleanpath, fullpath string, err error) {
	cleanpath = filepath.Clean(string(os.PathSeparator) + path)[1:]
	fullpath, err = symlink.FollowSymlinkInScope(filepath.Join(cs.root, path), cs.root)
	if err != nil {
		return "", "", fmt.Errorf("Forbidden path outside the context: %s (%s)", path, fullpath)
	}
	_, err = os.Lstat(fullpath)
	if err != nil {
		return "", "", convertPathError(err, path)
	}
	return
}

// Hash returns a hash for a single file in the source
func (cs *CachableSource) Hash(path string) (string, error) {
	n := cs.getRoot()
	sum := ""
	// TODO: check this for symlinks
	v, ok := n.Get([]byte(path))
	if !ok {
		sum = path
	} else {
		sum = v.(*fileInfo).sum
	}
	return sum, nil
}

// Root returns a root directory for the source
func (cs *CachableSource) Root() string {
	return cs.root
}

type fileInfo struct {
	sum string
}

func (fi *fileInfo) Hash() string {
	return fi.sum
}
