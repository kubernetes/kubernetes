package merkletrie

import (
	"fmt"
	"io"

	"github.com/go-git/go-git/v5/utils/merkletrie/noder"
)

// Action values represent the kind of things a Change can represent:
// insertion, deletions or modifications of files.
type Action int

// The set of possible actions in a change.
const (
	_ Action = iota
	Insert
	Delete
	Modify
)

// String returns the action as a human readable text.
func (a Action) String() string {
	switch a {
	case Insert:
		return "Insert"
	case Delete:
		return "Delete"
	case Modify:
		return "Modify"
	default:
		panic(fmt.Sprintf("unsupported action: %d", a))
	}
}

// A Change value represent how a noder has change between to merkletries.
type Change struct {
	// The noder before the change or nil if it was inserted.
	From noder.Path
	// The noder after the change or nil if it was deleted.
	To noder.Path
}

// Action is convenience method that returns what Action c represents.
func (c *Change) Action() (Action, error) {
	if c.From == nil && c.To == nil {
		return Action(0), fmt.Errorf("malformed change: nil from and to")
	}
	if c.From == nil {
		return Insert, nil
	}
	if c.To == nil {
		return Delete, nil
	}

	return Modify, nil
}

// NewInsert returns a new Change representing the insertion of n.
func NewInsert(n noder.Path) Change { return Change{To: n} }

// NewDelete returns a new Change representing the deletion of n.
func NewDelete(n noder.Path) Change { return Change{From: n} }

// NewModify returns a new Change representing that a has been modified and
// it is now b.
func NewModify(a, b noder.Path) Change {
	return Change{
		From: a,
		To:   b,
	}
}

// String returns a single change in human readable form, using the
// format: '<' + action + space + path + '>'.  The contents of the file
// before or after the change are not included in this format.
//
// Example: inserting a file at the path a/b/c.txt will return "<Insert
// a/b/c.txt>".
func (c Change) String() string {
	action, err := c.Action()
	if err != nil {
		panic(err)
	}

	var path string
	if action == Delete {
		path = c.From.String()
	} else {
		path = c.To.String()
	}

	return fmt.Sprintf("<%s %s>", action, path)
}

// Changes is a list of changes between to merkletries.
type Changes []Change

// NewChanges returns an empty list of changes.
func NewChanges() Changes {
	return Changes{}
}

// Add adds the change c to the list of changes.
func (l *Changes) Add(c Change) {
	*l = append(*l, c)
}

// AddRecursiveInsert adds the required changes to insert all the
// file-like noders found in root, recursively.
func (l *Changes) AddRecursiveInsert(root noder.Path) error {
	return l.addRecursive(root, NewInsert)
}

// AddRecursiveDelete adds the required changes to delete all the
// file-like noders found in root, recursively.
func (l *Changes) AddRecursiveDelete(root noder.Path) error {
	return l.addRecursive(root, NewDelete)
}

type noderToChangeFn func(noder.Path) Change // NewInsert or NewDelete

func (l *Changes) addRecursive(root noder.Path, ctor noderToChangeFn) error {
	if !root.IsDir() {
		l.Add(ctor(root))
		return nil
	}

	i, err := NewIterFromPath(root)
	if err != nil {
		return err
	}

	var current noder.Path
	for {
		if current, err = i.Step(); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		if current.IsDir() {
			continue
		}
		l.Add(ctor(current))
	}

	return nil
}
