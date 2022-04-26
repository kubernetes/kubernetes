package index

import (
	"bytes"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"
)

var (
	// ErrUnsupportedVersion is returned by Decode when the index file version
	// is not supported.
	ErrUnsupportedVersion = errors.New("unsupported version")
	// ErrEntryNotFound is returned by Index.Entry, if an entry is not found.
	ErrEntryNotFound = errors.New("entry not found")

	indexSignature              = []byte{'D', 'I', 'R', 'C'}
	treeExtSignature            = []byte{'T', 'R', 'E', 'E'}
	resolveUndoExtSignature     = []byte{'R', 'E', 'U', 'C'}
	endOfIndexEntryExtSignature = []byte{'E', 'O', 'I', 'E'}
)

// Stage during merge
type Stage int

const (
	// Merged is the default stage, fully merged
	Merged Stage = 1
	// AncestorMode is the base revision
	AncestorMode Stage = 1
	// OurMode is the first tree revision, ours
	OurMode Stage = 2
	// TheirMode is the second tree revision, theirs
	TheirMode Stage = 3
)

// Index contains the information about which objects are currently checked out
// in the worktree, having information about the working files. Changes in
// worktree are detected using this Index. The Index is also used during merges
type Index struct {
	// Version is index version
	Version uint32
	// Entries collection of entries represented by this Index. The order of
	// this collection is not guaranteed
	Entries []*Entry
	// Cache represents the 'Cached tree' extension
	Cache *Tree
	// ResolveUndo represents the 'Resolve undo' extension
	ResolveUndo *ResolveUndo
	// EndOfIndexEntry represents the 'End of Index Entry' extension
	EndOfIndexEntry *EndOfIndexEntry
}

// Add creates a new Entry and returns it. The caller should first check that
// another entry with the same path does not exist.
func (i *Index) Add(path string) *Entry {
	e := &Entry{
		Name: filepath.ToSlash(path),
	}

	i.Entries = append(i.Entries, e)
	return e
}

// Entry returns the entry that match the given path, if any.
func (i *Index) Entry(path string) (*Entry, error) {
	path = filepath.ToSlash(path)
	for _, e := range i.Entries {
		if e.Name == path {
			return e, nil
		}
	}

	return nil, ErrEntryNotFound
}

// Remove remove the entry that match the give path and returns deleted entry.
func (i *Index) Remove(path string) (*Entry, error) {
	path = filepath.ToSlash(path)
	for index, e := range i.Entries {
		if e.Name == path {
			i.Entries = append(i.Entries[:index], i.Entries[index+1:]...)
			return e, nil
		}
	}

	return nil, ErrEntryNotFound
}

// Glob returns the all entries matching pattern or nil if there is no matching
// entry. The syntax of patterns is the same as in filepath.Glob.
func (i *Index) Glob(pattern string) (matches []*Entry, err error) {
	pattern = filepath.ToSlash(pattern)
	for _, e := range i.Entries {
		m, err := match(pattern, e.Name)
		if err != nil {
			return nil, err
		}

		if m {
			matches = append(matches, e)
		}
	}

	return
}

// String is equivalent to `git ls-files --stage --debug`
func (i *Index) String() string {
	buf := bytes.NewBuffer(nil)
	for _, e := range i.Entries {
		buf.WriteString(e.String())
	}

	return buf.String()
}

// Entry represents a single file (or stage of a file) in the cache. An entry
// represents exactly one stage of a file. If a file path is unmerged then
// multiple Entry instances may appear for the same path name.
type Entry struct {
	// Hash is the SHA1 of the represented file
	Hash plumbing.Hash
	// Name is the  Entry path name relative to top level directory
	Name string
	// CreatedAt time when the tracked path was created
	CreatedAt time.Time
	// ModifiedAt time when the tracked path was changed
	ModifiedAt time.Time
	// Dev and Inode of the tracked path
	Dev, Inode uint32
	// Mode of the path
	Mode filemode.FileMode
	// UID and GID, userid and group id of the owner
	UID, GID uint32
	// Size is the length in bytes for regular files
	Size uint32
	// Stage on a merge is defines what stage is representing this entry
	// https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging
	Stage Stage
	// SkipWorktree used in sparse checkouts
	// https://git-scm.com/docs/git-read-tree#_sparse_checkout
	SkipWorktree bool
	// IntentToAdd record only the fact that the path will be added later
	// https://git-scm.com/docs/git-add ("git add -N")
	IntentToAdd bool
}

func (e Entry) String() string {
	buf := bytes.NewBuffer(nil)

	fmt.Fprintf(buf, "%06o %s %d\t%s\n", e.Mode, e.Hash, e.Stage, e.Name)
	fmt.Fprintf(buf, "  ctime: %d:%d\n", e.CreatedAt.Unix(), e.CreatedAt.Nanosecond())
	fmt.Fprintf(buf, "  mtime: %d:%d\n", e.ModifiedAt.Unix(), e.ModifiedAt.Nanosecond())
	fmt.Fprintf(buf, "  dev: %d\tino: %d\n", e.Dev, e.Inode)
	fmt.Fprintf(buf, "  uid: %d\tgid: %d\n", e.UID, e.GID)
	fmt.Fprintf(buf, "  size: %d\tflags: %x\n", e.Size, 0)

	return buf.String()
}

// Tree contains pre-computed hashes for trees that can be derived from the
// index. It helps speed up tree object generation from index for a new commit.
type Tree struct {
	Entries []TreeEntry
}

// TreeEntry entry of a cached Tree
type TreeEntry struct {
	// Path component (relative to its parent directory)
	Path string
	// Entries is the number of entries in the index that is covered by the tree
	// this entry represents.
	Entries int
	// Trees is the number that represents the number of subtrees this tree has
	Trees int
	// Hash object name for the object that would result from writing this span
	// of index as a tree.
	Hash plumbing.Hash
}

// ResolveUndo is used when a conflict is resolved (e.g. with "git add path"),
// these higher stage entries are removed and a stage-0 entry with proper
// resolution is added. When these higher stage entries are removed, they are
// saved in the resolve undo extension.
type ResolveUndo struct {
	Entries []ResolveUndoEntry
}

// ResolveUndoEntry contains the information about a conflict when is resolved
type ResolveUndoEntry struct {
	Path   string
	Stages map[Stage]plumbing.Hash
}

// EndOfIndexEntry is the End of Index Entry (EOIE) is used to locate the end of
// the variable length index entries and the beginning of the extensions. Code
// can take advantage of this to quickly locate the index extensions without
// having to parse through all of the index entries.
//
//  Because it must be able to be loaded before the variable length cache
//  entries and other index extensions, this extension must be written last.
type EndOfIndexEntry struct {
	// Offset to the end of the index entries
	Offset uint32
	// Hash is a SHA-1 over the extension types and their sizes (but not
	//	their contents).
	Hash plumbing.Hash
}
