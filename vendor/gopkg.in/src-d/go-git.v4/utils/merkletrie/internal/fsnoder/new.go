package fsnoder

import (
	"bytes"
	"fmt"
	"io"

	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"
)

// New function creates a full merkle trie from the string description of
// a filesystem tree.  See examples of the string format in the package
// description.
func New(s string) (noder.Noder, error) {
	return decodeDir([]byte(s), root)
}

const (
	root    = true
	nonRoot = false
)

// Expected data: a fsnoder description, for example: A(foo bar qux ...).
// When isRoot is true, unnamed dirs are supported, for example: (foo
// bar qux ...)
func decodeDir(data []byte, isRoot bool) (*dir, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil, io.EOF
	}

	// get the name of the dir and remove it from the data.  In case the
	// there is no name and isRoot is true, just use "" as the name.
	var name string
	switch end := bytes.IndexRune(data, dirStartMark); end {
	case -1:
		return nil, fmt.Errorf("%c not found", dirStartMark)
	case 0:
		if isRoot {
			name = ""
		} else {
			return nil, fmt.Errorf("inner unnamed dirs not allowed: %s", data)
		}
	default:
		name = string(data[0:end])
		data = data[end:]
	}

	// check data ends with the dirEndMark
	if data[len(data)-1] != dirEndMark {
		return nil, fmt.Errorf("malformed data: last %q not found",
			dirEndMark)
	}
	data = data[1 : len(data)-1] // remove initial '(' and last ')'

	children, err := decodeChildren(data)
	if err != nil {
		return nil, err
	}

	return newDir(name, children)
}

func isNumber(b rune) bool {
	return '0' <= b && b <= '9'
}

func isLetter(b rune) bool {
	return ('a' <= b && b <= 'z') || ('A' <= b && b <= 'Z')
}

func decodeChildren(data []byte) ([]noder.Noder, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil, nil
	}

	chunks := split(data)
	ret := make([]noder.Noder, len(chunks))
	var err error
	for i, c := range chunks {
		ret[i], err = decodeChild(c)
		if err != nil {
			return nil, fmt.Errorf("malformed element %d (%s): %s", i, c, err)
		}
	}

	return ret, nil
}

// returns the description of the elements of a dir.  It is just looking
// for spaces if they are not part of inner dirs.
func split(data []byte) [][]byte {
	chunks := [][]byte{}

	start := 0
	dirDepth := 0
	for i, b := range data {
		switch b {
		case dirStartMark:
			dirDepth++
		case dirEndMark:
			dirDepth--
		case dirElementSep:
			if dirDepth == 0 {
				chunks = append(chunks, data[start:i+1])
				start = i + 1
			}
		}
	}
	chunks = append(chunks, data[start:])

	return chunks
}

// A child can be a file or a dir.
func decodeChild(data []byte) (noder.Noder, error) {
	clean := bytes.TrimSpace(data)
	if len(data) < 3 {
		return nil, fmt.Errorf("element too short: %s", clean)
	}

	fileNameEnd := bytes.IndexRune(data, fileStartMark)
	dirNameEnd := bytes.IndexRune(data, dirStartMark)
	switch {
	case fileNameEnd == -1 && dirNameEnd == -1:
		return nil, fmt.Errorf(
			"malformed child, no file or dir start mark found")
	case fileNameEnd == -1:
		return decodeDir(clean, nonRoot)
	case dirNameEnd == -1:
		return decodeFile(clean)
	case dirNameEnd < fileNameEnd:
		return decodeDir(clean, nonRoot)
	case dirNameEnd > fileNameEnd:
		return decodeFile(clean)
	}

	return nil, fmt.Errorf("unreachable")
}

func decodeFile(data []byte) (noder.Noder, error) {
	nameEnd := bytes.IndexRune(data, fileStartMark)
	if nameEnd == -1 {
		return nil, fmt.Errorf("malformed file, no %c found", fileStartMark)
	}
	contentStart := nameEnd + 1
	contentEnd := bytes.IndexRune(data, fileEndMark)
	if contentEnd == -1 {
		return nil, fmt.Errorf("malformed file, no %c found", fileEndMark)
	}

	switch {
	case nameEnd > contentEnd:
		return nil, fmt.Errorf("malformed file, found %c before %c",
			fileEndMark, fileStartMark)
	case contentStart == contentEnd:
		name := string(data[:nameEnd])
		if !validFileName(name) {
			return nil, fmt.Errorf("invalid file name")
		}
		return newFile(name, "")
	default:
		name := string(data[:nameEnd])
		if !validFileName(name) {
			return nil, fmt.Errorf("invalid file name")
		}
		contents := string(data[contentStart:contentEnd])
		if !validFileContents(contents) {
			return nil, fmt.Errorf("invalid file contents")
		}
		return newFile(name, contents)
	}
}

func validFileName(s string) bool {
	for _, c := range s {
		if !isLetter(c) && c != '.' {
			return false
		}
	}

	return true
}

func validFileContents(s string) bool {
	for _, c := range s {
		if !isNumber(c) {
			return false
		}
	}

	return true
}

// HashEqual returns if a and b have the same hash.
func HashEqual(a, b noder.Hasher) bool {
	return bytes.Equal(a.Hash(), b.Hash())
}
