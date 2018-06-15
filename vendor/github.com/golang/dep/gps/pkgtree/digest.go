// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgtree

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"hash"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/pkg/errors"
)

const osPathSeparator = string(filepath.Separator)

// lineEndingReader is a `io.Reader` that converts CRLF sequences to LF.
//
// When cloning or checking out repositories, some Version Control Systems,
// VCSs, on some supported Go Operating System architectures, GOOS, will
// automatically convert line endings that end in a single line feed byte, LF,
// to line endings that end in a two byte sequence of carriage return, CR,
// followed by LF. This LF to CRLF conversion would cause otherwise identical
// versioned files to have different on disk contents simply based on which VCS
// and GOOS are involved. Different file contents for the same file would cause
// the resultant hashes to differ. In order to ensure file contents normalize
// and produce the same hash, this structure wraps an io.Reader that modifies
// the file's contents when it is read, translating all CRLF sequences to LF.
type lineEndingReader struct {
	src             io.Reader // source io.Reader from which this reads
	prevReadEndedCR bool      // used to track whether final byte of previous Read was CR
}

// newLineEndingReader returns a new lineEndingReader that reads from the
// specified source io.Reader.
func newLineEndingReader(src io.Reader) *lineEndingReader {
	return &lineEndingReader{src: src}
}

var crlf = []byte("\r\n")

// Read consumes bytes from the structure's source io.Reader to fill the
// specified slice of bytes. It converts all CRLF byte sequences to LF, and
// handles cases where CR and LF straddle across two Read operations.
func (f *lineEndingReader) Read(buf []byte) (int, error) {
	buflen := len(buf)
	if f.prevReadEndedCR {
		// Read one fewer bytes so we have room if the first byte of the
		// upcoming Read is not a LF, in which case we will need to insert
		// trailing CR from previous read.
		buflen--
	}
	nr, er := f.src.Read(buf[:buflen])
	if nr > 0 {
		if f.prevReadEndedCR && buf[0] != '\n' {
			// Having a CRLF split across two Read operations is rare, so the
			// performance impact of copying entire buffer to the right by one
			// byte, while suboptimal, will at least will not happen very
			// often. This negative performance impact is mitigated somewhat on
			// many Go compilation architectures, GOARCH, because the `copy`
			// builtin uses a machine opcode for performing the memory copy on
			// possibly overlapping regions of memory. This machine opcodes is
			// not instantaneous and does require multiple CPU cycles to
			// complete, but is significantly faster than the application
			// looping through bytes.
			copy(buf[1:nr+1], buf[:nr]) // shift data to right one byte
			buf[0] = '\r'               // insert the previous skipped CR byte at start of buf
			nr++                        // pretend we read one more byte
		}

		// Remove any CRLF sequences in the buffer using `bytes.Index` because,
		// like the `copy` builtin on many GOARCHs, it also takes advantage of a
		// machine opcode to search for byte patterns.
		var searchOffset int // index within buffer from whence the search will commence for each loop; set to the index of the end of the previous loop.
		var shiftCount int   // each subsequenct shift operation needs to shift bytes to the left by one more position than the shift that preceded it.
		previousIndex := -1  // index of previously found CRLF; -1 means no previous index
		for {
			index := bytes.Index(buf[searchOffset:nr], crlf)
			if index == -1 {
				break
			}
			index += searchOffset // convert relative index to absolute
			if previousIndex != -1 {
				// shift substring between previous index and this index
				copy(buf[previousIndex-shiftCount:], buf[previousIndex+1:index])
				shiftCount++ // next shift needs to be 1 byte to the left
			}
			previousIndex = index
			searchOffset = index + 2 // start next search after len(crlf)
		}
		if previousIndex != -1 {
			// handle final shift
			copy(buf[previousIndex-shiftCount:], buf[previousIndex+1:nr])
			shiftCount++
		}
		nr -= shiftCount // shorten byte read count by number of shifts executed

		// When final byte from a read operation is CR, do not emit it until
		// ensure first byte on next read is not LF.
		if f.prevReadEndedCR = buf[nr-1] == '\r'; f.prevReadEndedCR {
			nr-- // pretend byte was never read from source
		}
	} else if f.prevReadEndedCR {
		// Reading from source returned nothing, but this struct is sitting on a
		// trailing CR from previous Read, so let's give it to client now.
		buf[0] = '\r'
		nr = 1
		er = nil
		f.prevReadEndedCR = false // prevent infinite loop
	}
	return nr, er
}

// writeBytesWithNull appends the specified data to the specified hash, followed by
// the NULL byte, in order to make accidental hash collisions less likely.
func writeBytesWithNull(h hash.Hash, data []byte) {
	// Ignore return values from writing to the hash, because hash write always
	// returns nil error.
	_, _ = h.Write(append(data, 0))
}

// dirWalkClosure is used to reduce number of allocation involved in closing
// over these variables.
type dirWalkClosure struct {
	someCopyBufer []byte // allocate once and reuse for each file copy
	someModeBytes []byte // allocate once and reuse for each node
	someDirLen    int
	someHash      hash.Hash
}

// DigestFromDirectory returns a hash of the specified directory contents, which
// will match the hash computed for any directory on any supported Go platform
// whose contents exactly match the specified directory.
//
// This function ignores any file system node named `vendor`, `.bzr`, `.git`,
// `.hg`, and `.svn`, as these are typically used as Version Control System
// (VCS) directories.
//
// Other than the `vendor` and VCS directories mentioned above, the calculated
// hash includes the pathname to every discovered file system node, whether it
// is an empty directory, a non-empty directory, empty file, non-empty file, or
// symbolic link. If a symbolic link, the referent name is included. If a
// non-empty file, the file's contents are included. If a non-empty directory,
// the contents of the directory are included.
//
// While filepath.Walk could have been used, that standard library function
// skips symbolic links, and for now, we want the hash to include the symbolic
// link referents.
func DigestFromDirectory(osDirname string) ([]byte, error) {
	osDirname = filepath.Clean(osDirname)

	// Create a single hash instance for the entire operation, rather than a new
	// hash for each node we encounter.

	closure := dirWalkClosure{
		someCopyBufer: make([]byte, 4*1024), // only allocate a single page
		someModeBytes: make([]byte, 4),      // scratch place to store encoded os.FileMode (uint32)
		someDirLen:    len(osDirname) + len(osPathSeparator),
		someHash:      sha256.New(),
	}

	err := DirWalk(osDirname, func(osPathname string, info os.FileInfo, err error) error {
		if err != nil {
			return err // DirWalk received an error during initial Lstat
		}

		var osRelative string
		if len(osPathname) > closure.someDirLen {
			osRelative = osPathname[closure.someDirLen:]
		}

		switch filepath.Base(osRelative) {
		case "vendor", ".bzr", ".git", ".hg", ".svn":
			return filepath.SkipDir
		}

		// We could make our own enum-like data type for encoding the file type,
		// but Go's runtime already gives us architecture independent file
		// modes, as discussed in `os/types.go`:
		//
		//    Go's runtime FileMode type has same definition on all systems, so
		//    that information about files can be moved from one system to
		//    another portably.
		var mt os.FileMode

		// We only care about the bits that identify the type of a file system
		// node, and can ignore append, exclusive, temporary, setuid, setgid,
		// permission bits, and sticky bits, which are coincident to bits which
		// declare type of the file system node.
		modeType := info.Mode() & os.ModeType
		var shouldSkip bool // skip some types of file system nodes

		switch {
		case modeType&os.ModeDir > 0:
			mt = os.ModeDir
			// DirWalkFunc itself does not need to enumerate children, because
			// DirWalk will do that for us.
			shouldSkip = true
		case modeType&os.ModeSymlink > 0:
			mt = os.ModeSymlink
		case modeType&os.ModeNamedPipe > 0:
			mt = os.ModeNamedPipe
			shouldSkip = true
		case modeType&os.ModeSocket > 0:
			mt = os.ModeSocket
			shouldSkip = true
		case modeType&os.ModeDevice > 0:
			mt = os.ModeDevice
			shouldSkip = true
		}

		// Write the relative pathname to hash because the hash is a function of
		// the node names, node types, and node contents. Added benefit is that
		// empty directories, named pipes, sockets, devices, and symbolic links
		// will also affect final hash value. Use `filepath.ToSlash` to ensure
		// relative pathname is os-agnostic.
		writeBytesWithNull(closure.someHash, []byte(filepath.ToSlash(osRelative)))

		binary.LittleEndian.PutUint32(closure.someModeBytes, uint32(mt)) // encode the type of mode
		writeBytesWithNull(closure.someHash, closure.someModeBytes)      // and write to hash

		if shouldSkip {
			return nil // nothing more to do for some of the node types
		}

		if mt == os.ModeSymlink { // okay to check for equivalence because we set to this value
			osRelative, err = os.Readlink(osPathname) // read the symlink referent
			if err != nil {
				return errors.Wrap(err, "cannot Readlink")
			}
			writeBytesWithNull(closure.someHash, []byte(filepath.ToSlash(osRelative))) // write referent to hash
			return nil                                                                 // proceed to next node in queue
		}

		// If we get here, node is a regular file.
		fh, err := os.Open(osPathname)
		if err != nil {
			return errors.Wrap(err, "cannot Open")
		}

		var bytesWritten int64
		bytesWritten, err = io.CopyBuffer(closure.someHash, newLineEndingReader(fh), closure.someCopyBufer) // fast copy of file contents to hash
		err = errors.Wrap(err, "cannot Copy")                                                               // errors.Wrap only wraps non-nil, so skip extra check
		writeBytesWithNull(closure.someHash, []byte(strconv.FormatInt(bytesWritten, 10)))                   // 10: format file size as base 10 integer

		// Close the file handle to the open file without masking
		// possible previous error value.
		if er := fh.Close(); err == nil {
			err = errors.Wrap(er, "cannot Close")
		}
		return err
	})
	if err != nil {
		return nil, err
	}
	return closure.someHash.Sum(nil), nil
}

// VendorStatus represents one of a handful of possible status conditions for a
// particular file system node in the vendor directory tree.
type VendorStatus uint8

const (
	// NotInLock is used when a file system node exists for which there is no
	// corresponding dependency in the lock file.
	NotInLock VendorStatus = iota

	// NotInTree is used when a lock file dependency exists for which there is
	// no corresponding file system node.
	NotInTree

	// NoMismatch is used when the digest for a dependency listed in the
	// lockfile matches what is calculated from the file system.
	NoMismatch

	// EmptyDigestInLock is used when the digest for a dependency listed in the
	// lock file is the empty string. While this is a special case of
	// DigestMismatchInLock, keeping both cases discrete is a desired feature.
	EmptyDigestInLock

	// DigestMismatchInLock is used when the digest for a dependency listed in
	// the lock file does not match what is calculated from the file system.
	DigestMismatchInLock
)

func (ls VendorStatus) String() string {
	switch ls {
	case NotInTree:
		return "not in tree"
	case NotInLock:
		return "not in lock"
	case NoMismatch:
		return "match"
	case EmptyDigestInLock:
		return "empty digest in lock"
	case DigestMismatchInLock:
		return "mismatch"
	}
	return "unknown"
}

// fsnode is used to track which file system nodes are required by the lock
// file. When a directory is found whose name matches one of the declared
// projects in the lock file, e.g., "github.com/alice/alice1", an fsnode is
// created for that directory, but not for any of its children. All other file
// system nodes encountered will result in a fsnode created to represent it.
type fsnode struct {
	osRelative           string // os-specific relative path of a resource under vendor root
	isRequiredAncestor   bool   // true iff this node or one of its descendants is in the lock file
	myIndex, parentIndex int    // index of this node and its parent in the tree's slice
}

// VerifyDepTree verifies a dependency tree according to expected digest sums,
// and returns an associative array of file system nodes and their respective
// vendor status conditions.
//
// The keys to the expected digest sums associative array represent the
// project's dependencies, and each is required to be expressed using the
// solidus character, `/`, as its path separator. For example, even on a GOOS
// platform where the file system path separator is a character other than
// solidus, one particular dependency would be represented as
// "github.com/alice/alice1".
func VerifyDepTree(osDirname string, wantSums map[string][]byte) (map[string]VendorStatus, error) {
	osDirname = filepath.Clean(osDirname)

	// Ensure top level pathname is a directory
	fi, err := os.Stat(osDirname)
	if err != nil {
		return nil, errors.Wrap(err, "cannot Stat")
	}
	if !fi.IsDir() {
		return nil, errors.Errorf("cannot verify non directory: %q", osDirname)
	}

	// Initialize work queue with a node representing the specified directory
	// name by declaring its relative pathname under the directory name as the
	// empty string.
	currentNode := &fsnode{osRelative: "", parentIndex: -1, isRequiredAncestor: true}
	queue := []*fsnode{currentNode} // queue of directories that must be inspected

	// In order to identify all file system nodes that are not in the lock file,
	// represented by the specified expected sums parameter, and in order to
	// only report the top level of a subdirectory of file system nodes, rather
	// than every node internal to them, we will create a tree of nodes stored
	// in a slice. We do this because we cannot predict the depth at which
	// project roots occur. Some projects are fewer than and some projects more
	// than the typical three layer subdirectory under the vendor root
	// directory.
	//
	// For a following few examples, assume the below vendor root directory:
	//
	// github.com/alice/alice1/a1.go
	// github.com/alice/alice2/a2.go
	// github.com/bob/bob1/b1.go
	// github.com/bob/bob2/b2.go
	// launchpad.net/nifty/n1.go
	//
	// 1) If only the `alice1` and `alice2` projects were in the lock file, we'd
	// prefer the output to state that `github.com/bob` is `NotInLock`, and
	// `launchpad.net/nifty` is `NotInLock`.
	//
	// 2) If `alice1`, `alice2`, and `bob1` were in the lock file, we'd want to
	// report `github.com/bob/bob2` as `NotInLock`, and `launchpad.net/nifty` is
	// `NotInLock`.
	//
	// 3) If none of `alice1`, `alice2`, `bob1`, or `bob2` were in the lock
	// file, the entire `github.com` directory would be reported as `NotInLock`,
	// along with `launchpad.net/nifty` is `NotInLock`.
	//
	// Each node in our tree has the slice index of its parent node, so once we
	// can categorically state a particular directory is required because it is
	// in the lock file, we can mark all of its ancestors as also being
	// required. Then, when we finish walking the directory hierarchy, any nodes
	// which are not required but have a required parent will be marked as
	// `NotInLock`.
	nodes := []*fsnode{currentNode}

	// Create associative array to store the results of calling this function.
	slashStatus := make(map[string]VendorStatus)

	// Mark directories of expected projects as required. When each respective
	// project is later found while traversing the vendor root hierarchy, its
	// status will be updated to reflect whether its digest is empty, or,
	// whether or not it matches the expected digest.
	for slashPathname := range wantSums {
		slashStatus[slashPathname] = NotInTree
	}

	for len(queue) > 0 {
		// Pop node from the top of queue (depth first traversal, reverse
		// lexicographical order inside a directory), clearing the value stored
		// in the slice's backing array as we proceed.
		lq1 := len(queue) - 1
		currentNode, queue[lq1], queue = queue[lq1], nil, queue[:lq1]
		slashPathname := filepath.ToSlash(currentNode.osRelative)
		osPathname := filepath.Join(osDirname, currentNode.osRelative)

		if expectedSum, ok := wantSums[slashPathname]; ok {
			ls := EmptyDigestInLock
			if len(expectedSum) > 0 {
				projectSum, err := DigestFromDirectory(osPathname)
				if err != nil {
					return nil, errors.Wrap(err, "cannot compute dependency hash")
				}
				if bytes.Equal(projectSum, expectedSum) {
					ls = NoMismatch
				} else {
					ls = DigestMismatchInLock
				}
			}
			slashStatus[slashPathname] = ls

			// Mark current nodes and all its parents as required.
			for i := currentNode.myIndex; i != -1; i = nodes[i].parentIndex {
				nodes[i].isRequiredAncestor = true
			}

			// Do not need to process this directory's contents because we
			// already accounted for its contents while calculating its digest.
			continue
		}

		osChildrenNames, err := sortedChildrenFromDirname(osPathname)
		if err != nil {
			return nil, errors.Wrap(err, "cannot get sorted list of directory children")
		}
		for _, osChildName := range osChildrenNames {
			switch osChildName {
			case ".", "..", "vendor", ".bzr", ".git", ".hg", ".svn":
				// skip
			default:
				osChildRelative := filepath.Join(currentNode.osRelative, osChildName)
				osChildPathname := filepath.Join(osDirname, osChildRelative)

				// Create a new fsnode for this file system node, with a parent
				// index set to the index of the current node.
				otherNode := &fsnode{osRelative: osChildRelative, myIndex: len(nodes), parentIndex: currentNode.myIndex}

				fi, err := os.Stat(osChildPathname)
				if err != nil {
					return nil, errors.Wrap(err, "cannot Stat")
				}
				nodes = append(nodes, otherNode) // Track all file system nodes...
				if fi.IsDir() {
					queue = append(queue, otherNode) // but only need to add directories to the work queue.
				}
			}
		}
	}

	// Ignoring first node in the list, walk nodes from last to first. Whenever
	// the current node is not required, but its parent is required, then the
	// current node ought to be marked as `NotInLock`.
	for len(nodes) > 1 {
		// Pop node from top of queue, clearing the value stored in the slice's
		// backing array as we proceed.
		ln1 := len(nodes) - 1
		currentNode, nodes[ln1], nodes = nodes[ln1], nil, nodes[:ln1]

		if !currentNode.isRequiredAncestor && nodes[currentNode.parentIndex].isRequiredAncestor {
			slashStatus[filepath.ToSlash(currentNode.osRelative)] = NotInLock
		}
	}
	currentNode, nodes = nil, nil

	return slashStatus, nil
}
