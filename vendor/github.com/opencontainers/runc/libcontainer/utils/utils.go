package utils

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"
)

const (
	exitSignalOffset = 128
)

// ExitStatus returns the correct exit status for a process based on if it
// was signaled or exited cleanly
func ExitStatus(status unix.WaitStatus) int {
	if status.Signaled() {
		return exitSignalOffset + int(status.Signal())
	}
	return status.ExitStatus()
}

// WriteJSON writes the provided struct v to w using standard json marshaling
// without a trailing newline. This is used instead of json.Encoder because
// there might be a problem in json decoder in some cases, see:
// https://github.com/docker/docker/issues/14203#issuecomment-174177790
func WriteJSON(w io.Writer, v interface{}) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// CleanPath makes a path safe for use with filepath.Join. This is done by not
// only cleaning the path, but also (if the path is relative) adding a leading
// '/' and cleaning it (then removing the leading '/'). This ensures that a
// path resulting from prepending another path will always resolve to lexically
// be a subdirectory of the prefixed path. This is all done lexically, so paths
// that include symlinks won't be safe as a result of using CleanPath.
func CleanPath(path string) string {
	// Deal with empty strings nicely.
	if path == "" {
		return ""
	}

	// Ensure that all paths are cleaned (especially problematic ones like
	// "/../../../../../" which can cause lots of issues).
	path = filepath.Clean(path)

	// If the path isn't absolute, we need to do more processing to fix paths
	// such as "../../../../<etc>/some/path". We also shouldn't convert absolute
	// paths to relative ones.
	if !filepath.IsAbs(path) {
		path = filepath.Clean(string(os.PathSeparator) + path)
		// This can't fail, as (by definition) all paths are relative to root.
		path, _ = filepath.Rel(string(os.PathSeparator), path)
	}

	// Clean the path again for good measure.
	return filepath.Clean(path)
}

// stripRoot returns the passed path, stripping the root path if it was
// (lexicially) inside it. Note that both passed paths will always be treated
// as absolute, and the returned path will also always be absolute. In
// addition, the paths are cleaned before stripping the root.
func stripRoot(root, path string) string {
	// Make the paths clean and absolute.
	root, path = CleanPath("/"+root), CleanPath("/"+path)
	switch {
	case path == root:
		path = "/"
	case root == "/":
		// do nothing
	case strings.HasPrefix(path, root+"/"):
		path = strings.TrimPrefix(path, root+"/")
	}
	return CleanPath("/" + path)
}

// SearchLabels searches through a list of key=value pairs for a given key,
// returning its value, and the binary flag telling whether the key exist.
func SearchLabels(labels []string, key string) (string, bool) {
	key += "="
	for _, s := range labels {
		if strings.HasPrefix(s, key) {
			return s[len(key):], true
		}
	}
	return "", false
}

// Annotations returns the bundle path and user defined annotations from the
// libcontainer state.  We need to remove the bundle because that is a label
// added by libcontainer.
func Annotations(labels []string) (bundle string, userAnnotations map[string]string) {
	userAnnotations = make(map[string]string)
	for _, l := range labels {
		name, value, ok := strings.Cut(l, "=")
		if !ok {
			continue
		}
		if name == "bundle" {
			bundle = value
		} else {
			userAnnotations[name] = value
		}
	}
	return
}
