package nuts

/*
Path Prefix Scans

The prefix scanning methods `SeekPathConflict` and `SeekPathMatch` facilitate maintenance and access to buckets of paths
supporting variable elements with exclusive matches.  Paths are `/` delimited, must begin with a `/`, and elements
beginning with `:` or `*` are variable.

Examples:
	/
	/blogs/
	/blogs/:blog_id

Variable Paths

Path elements beginning with a `:` match any single element.  Path elements beginning with `*` match any remaining
suffix, and therefore must be the last element.

Examples:
	Path:  /blogs/:blog_id
	Match: /blogs/someblog

	Path:  /blogs/:blog_id/comments/:comment_id/*suffix
	Match: /blogs/42/comments/100/edit

Exclusive Matches

Using `SeekPathConflict` before putting new paths to ensure the bucket remains conflict-free guarantees that
`SeekPathMatch` will never match more than one path.

Examples:
	Conflicts: /blogs/:blog_id, /blogs/golang
	Match:	   /blogs/golang

	Conflicts: /blogs/*, /blogs/:blog_id/comments
	Match:     /blogs/42/comments
*/

import (
	"bytes"

	"github.com/boltdb/bolt"
)

// SeekPathMatch seeks an entry which matches `path`, or returns `nil, nil` when no match is found.
// Returned key may be `path`, or a matching dynamic path.
// Matches are exclusive if the set of keys are conflict free (see SeekPathConflict).
func SeekPathMatch(c *bolt.Cursor, path []byte) ([]byte, []byte) {
	// Validation
	if len(path) == 0 {
		return nil, nil
	}
	if path[0] != '/' {
		return nil, nil
	}

	// Exact match fast-path
	if k, v := c.Seek(path); bytes.Equal(k, path) {
		return k, v
	}

	// Prefix scan
	prefixBuf := bytes.NewBuffer(make([]byte, 0, len(path)))
	for {
		// Match slash
		prefixBuf.WriteByte('/')
		prefix := prefixBuf.Bytes()
		k, v := c.Seek(prefix)
		if !bytes.HasPrefix(k, prefix) {
			return nil, nil
		}
		// Advance past '/'
		path = path[1:]

		// Exact match required for trailing slash.
		if len(path) == 0 {
			if len(k) == len(prefix) {
				return k, v
			}
			return nil, nil
		}

		// Advance cursor past exact match to first prefix match.
		if len(k) == len(prefix) {
			k, v = c.Next()
			if !bytes.HasPrefix(k, prefix) {
				return nil, nil
			}
		}

		// Find end of element.
		i := bytes.IndexByte(path, '/')
		last := i < 0

		switch k[len(prefix)] {
		case '*':
			return k, v

		case ':':
			// Append variable path element to prefix
			ki := bytes.IndexByte(k[len(prefix):], '/')
			if ki < 0 {
				prefixBuf.Write(k[len(prefix):])
			} else {
				prefixBuf.Write(k[len(prefix) : len(prefix)+ki])
			}

			if last {
				// Exact match required for last element.
				prefix = prefixBuf.Bytes()
				if k, v = c.Seek(prefix); bytes.Equal(k, prefix) {
					return k, v
				}
				return nil, nil
			}

		default:
			// Append path component to prefix.
			if last {
				prefixBuf.Write(path)
			} else {
				prefixBuf.Write(path[:i])
			}

			prefix = prefixBuf.Bytes()
			k, v = c.Seek(prefix)

			if last {
				// Exact match required for last element.
				if bytes.Equal(k, prefix) {
					return k, v
				}
				return nil, nil
			}

			// Prefix match required for other elements.
			if !bytes.HasPrefix(k, prefix) {
				return nil, nil
			}
		}

		// Advance past element.
		path = path[i:]
	}
}

// SeekPathConflict seeks an entry which conflicts with `path`, and returns the first encountered or `nil, nil` if none
// is found.
func SeekPathConflict(c *bolt.Cursor, path []byte) ([]byte, []byte) {
	// Validation
	if len(path) == 0 {
		return nil, nil
	}
	if path[0] != '/' {
		return nil, nil
	}

	// Fast-path for exact and prefix match.
	if k, v := c.Seek(path); bytes.Equal(k, path) {
		return k, v
	} else if bytes.HasPrefix(k, path) {
		// Any prefixed k is good enough when path ends in '/'.
		if path[len(path)-1] == '/' {
			return nil, nil
		}

		// If k's last element is longer it could be a conflict.
		if k[len(path)] == '/' {
			return nil, nil
		}
	}

	// Prefix scan.
	i := 0
	for {
		i++

		// Match slash.
		prefix := path[:i]
		k, v := c.Seek(prefix)
		if !bytes.HasPrefix(k, prefix) {
			return nil, nil
		}

		// Exact match is a conflict for trailing slash.
		if i == len(path) {
			if len(k) == len(path) {
				return k, v
			}
			return nil, nil
		}

		// Advance cursor past exact match to first prefix match.
		if len(k) == len(prefix) {
			k, v = c.Next()
			if !bytes.HasPrefix(k, prefix) {
				return nil, nil
			}
		}

		// Find end of element.
		offset := bytes.IndexByte(path[i:], '/')
		last := offset < 0
		if last {
			i = len(path)
		} else {
			i += offset
		}

		switch k[len(prefix)] {
		case '*':
			return k, v

		case ':':
			// Find end of element.
			kPrefix := k
			offset := bytes.IndexByte(k[len(prefix):], '/')
			if offset > 0 {
				kPrefix = k[:len(prefix)+offset]
			}

			// Exact match required through variable element.
			prefix = path[:i]
			if !bytes.Equal(prefix, kPrefix) {
				return k, v
			}

			if last {
				// Exact match is a conflict for the last element.
				if k, v = c.Seek(prefix); bytes.Equal(k, prefix) {
					return k, v
				}
				return nil, nil
			}

		default:
			// Static (non-variable) element required.
			next := path[len(prefix)]
			if next == ':' || next == '*' {
				return k, v
			}

			prefix = path[:i]
			k, v = c.Seek(prefix)

			if last {
				// Exact match is a conflict for the last element.
				if bytes.Equal(k, prefix) {
					return k, v
				}
				return nil, nil
			}

			if !bytes.HasPrefix(k, prefix) {
				return nil, nil
			}
		}
	}
}
