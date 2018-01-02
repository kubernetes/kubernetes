package object

import (
	"bytes"
	"fmt"
	"strings"

	"gopkg.in/src-d/go-git.v4/utils/merkletrie"
)

// Change values represent a detected change between two git trees.  For
// modifications, From is the original status of the node and To is its
// final status.  For insertions, From is the zero value and for
// deletions To is the zero value.
type Change struct {
	From ChangeEntry
	To   ChangeEntry
}

var empty = ChangeEntry{}

// Action returns the kind of action represented by the change, an
// insertion, a deletion or a modification.
func (c *Change) Action() (merkletrie.Action, error) {
	if c.From == empty && c.To == empty {
		return merkletrie.Action(0),
			fmt.Errorf("malformed change: empty from and to")
	}
	if c.From == empty {
		return merkletrie.Insert, nil
	}
	if c.To == empty {
		return merkletrie.Delete, nil
	}

	return merkletrie.Modify, nil
}

// Files return the files before and after a change.
// For insertions from will be nil. For deletions to will be nil.
func (c *Change) Files() (from, to *File, err error) {
	action, err := c.Action()
	if err != nil {
		return
	}

	if action == merkletrie.Insert || action == merkletrie.Modify {
		to, err = c.To.Tree.TreeEntryFile(&c.To.TreeEntry)
		if !c.To.TreeEntry.Mode.IsFile() {
			return nil, nil, nil
		}

		if err != nil {
			return
		}
	}

	if action == merkletrie.Delete || action == merkletrie.Modify {
		from, err = c.From.Tree.TreeEntryFile(&c.From.TreeEntry)
		if !c.From.TreeEntry.Mode.IsFile() {
			return nil, nil, nil
		}

		if err != nil {
			return
		}
	}

	return
}

func (c *Change) String() string {
	action, err := c.Action()
	if err != nil {
		return fmt.Sprintf("malformed change")
	}

	return fmt.Sprintf("<Action: %s, Path: %s>", action, c.name())
}

// Patch returns a Patch with all the file changes in chunks. This
// representation can be used to create several diff outputs.
func (c *Change) Patch() (*Patch, error) {
	return getPatch("", c)
}

func (c *Change) name() string {
	if c.From != empty {
		return c.From.Name
	}

	return c.To.Name
}

// ChangeEntry values represent a node that has suffered a change.
type ChangeEntry struct {
	// Full path of the node using "/" as separator.
	Name string
	// Parent tree of the node that has changed.
	Tree *Tree
	// The entry of the node.
	TreeEntry TreeEntry
}

// Changes represents a collection of changes between two git trees.
// Implements sort.Interface lexicographically over the path of the
// changed files.
type Changes []*Change

func (c Changes) Len() int {
	return len(c)
}

func (c Changes) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}

func (c Changes) Less(i, j int) bool {
	return strings.Compare(c[i].name(), c[j].name()) < 0
}

func (c Changes) String() string {
	var buffer bytes.Buffer
	buffer.WriteString("[")
	comma := ""
	for _, v := range c {
		buffer.WriteString(comma)
		buffer.WriteString(v.String())
		comma = ", "
	}
	buffer.WriteString("]")

	return buffer.String()
}

// Patch returns a Patch with all the changes in chunks. This
// representation can be used to create several diff outputs.
func (c Changes) Patch() (*Patch, error) {
	return getPatch("", c...)
}
