package diff

import (
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"
)

// Operation defines the operation of a diff item.
type Operation int

const (
	// Equal item represents a equals diff.
	Equal Operation = iota
	// Add item represents an insert diff.
	Add
	// Delete item represents a delete diff.
	Delete
)

// Patch represents a collection of steps to transform several files.
type Patch interface {
	// FilePatches returns a slice of patches per file.
	FilePatches() []FilePatch
	// Message returns an optional message that can be at the top of the
	// Patch representation.
	Message() string
}

// FilePatch represents the necessary steps to transform one file to another.
type FilePatch interface {
	// IsBinary returns true if this patch is representing a binary file.
	IsBinary() bool
	// Files returns the from and to Files, with all the necessary metadata to
	// about them. If the patch creates a new file, "from" will be nil.
	// If the patch deletes a file, "to" will be nil.
	Files() (from, to File)
	// Chunks returns a slice of ordered changes to transform "from" File to
	// "to" File. If the file is a binary one, Chunks will be empty.
	Chunks() []Chunk
}

// File contains all the file metadata necessary to print some patch formats.
type File interface {
	// Hash returns the File Hash.
	Hash() plumbing.Hash
	// Mode returns the FileMode.
	Mode() filemode.FileMode
	// Path returns the complete Path to the file, including the filename.
	Path() string
}

// Chunk represents a portion of a file transformation to another.
type Chunk interface {
	// Content contains the portion of the file.
	Content() string
	// Type contains the Operation to do with this Chunk.
	Type() Operation
}
