package noder

import (
	"bytes"
	"strings"
)

// Path values represent a noder and its ancestors.  The root goes first
// and the actual final noder the path is referring to will be the last.
//
// A path implements the Noder interface, redirecting all the interface
// calls to its final noder.
//
// Paths build from an empty Noder slice are not valid paths and should
// not be used.
type Path []Noder

// String returns the full path of the final noder as a string, using
// "/" as the separator.
func (p Path) String() string {
	var buf bytes.Buffer
	sep := ""
	for _, e := range p {
		_, _ = buf.WriteString(sep)
		sep = "/"
		_, _ = buf.WriteString(e.Name())
	}

	return buf.String()
}

// Last returns the final noder in the path.
func (p Path) Last() Noder {
	return p[len(p)-1]
}

// Hash returns the hash of the final noder of the path.
func (p Path) Hash() []byte {
	return p.Last().Hash()
}

// Name returns the name of the final noder of the path.
func (p Path) Name() string {
	return p.Last().Name()
}

// IsDir returns if the final noder of the path is a directory-like
// noder.
func (p Path) IsDir() bool {
	return p.Last().IsDir()
}

// Children returns the children of the final noder in the path.
func (p Path) Children() ([]Noder, error) {
	return p.Last().Children()
}

// NumChildren returns the number of children the final noder of the
// path has.
func (p Path) NumChildren() (int, error) {
	return p.Last().NumChildren()
}

// Compare returns -1, 0 or 1 if the path p is smaller, equal or bigger
// than other, in "directory order"; for example:
//
// "a" < "b"
// "a/b/c/d/z" < "b"
// "a/b/a" > "a/b"
func (p Path) Compare(other Path) int {
	i := 0
	for {
		switch {
		case len(other) == len(p) && i == len(p):
			return 0
		case i == len(other):
			return 1
		case i == len(p):
			return -1
		default:
			// We do *not* normalize Unicode here. CGit doesn't.
			// https://github.com/src-d/go-git/issues/1057
			cmp := strings.Compare(p[i].Name(), other[i].Name())
			if cmp != 0 {
				return cmp
			}
		}
		i++
	}
}
