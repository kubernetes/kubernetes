package utils

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unsafe"

	"github.com/cyphar/filepath-securejoin"
	"golang.org/x/sys/unix"
)

const (
	exitSignalOffset = 128
)

// NativeEndian is the native byte order of the host system.
var NativeEndian binary.ByteOrder

func init() {
	// Copied from <golang.org/x/net/internal/socket/sys.go>.
	i := uint32(1)
	b := (*[4]byte)(unsafe.Pointer(&i))
	if b[0] == 1 {
		NativeEndian = binary.LittleEndian
	} else {
		NativeEndian = binary.BigEndian
	}
}

// ResolveRootfs ensures that the current working directory is
// not a symlink and returns the absolute path to the rootfs
func ResolveRootfs(uncleanRootfs string) (string, error) {
	rootfs, err := filepath.Abs(uncleanRootfs)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(rootfs)
}

// ExitStatus returns the correct exit status for a process based on if it
// was signaled or exited cleanly
func ExitStatus(status unix.WaitStatus) int {
	if status.Signaled() {
		return exitSignalOffset + int(status.Signal())
	}
	return status.ExitStatus()
}

// WriteJSON writes the provided struct v to w using standard json marshaling
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

// WithProcfd runs the passed closure with a procfd path (/proc/self/fd/...)
// corresponding to the unsafePath resolved within the root. Before passing the
// fd, this path is verified to have been inside the root -- so operating on it
// through the passed fdpath should be safe. Do not access this path through
// the original path strings, and do not attempt to use the pathname outside of
// the passed closure (the file handle will be freed once the closure returns).
func WithProcfd(root, unsafePath string, fn func(procfd string) error) error {
	// Remove the root then forcefully resolve inside the root.
	unsafePath = stripRoot(root, unsafePath)
	path, err := securejoin.SecureJoin(root, unsafePath)
	if err != nil {
		return fmt.Errorf("resolving path inside rootfs failed: %v", err)
	}

	// Open the target path.
	fh, err := os.OpenFile(path, unix.O_PATH|unix.O_CLOEXEC, 0)
	if err != nil {
		return fmt.Errorf("open o_path procfd: %w", err)
	}
	defer fh.Close()

	// Double-check the path is the one we expected.
	procfd := "/proc/self/fd/" + strconv.Itoa(int(fh.Fd()))
	if realpath, err := os.Readlink(procfd); err != nil {
		return fmt.Errorf("procfd verification failed: %w", err)
	} else if realpath != path {
		return fmt.Errorf("possibly malicious path detected -- refusing to operate on %s", realpath)
	}

	// Run the closure.
	return fn(procfd)
}

// SearchLabels searches a list of key-value pairs for the provided key and
// returns the corresponding value. The pairs must be separated with '='.
func SearchLabels(labels []string, query string) string {
	for _, l := range labels {
		parts := strings.SplitN(l, "=", 2)
		if len(parts) < 2 {
			continue
		}
		if parts[0] == query {
			return parts[1]
		}
	}
	return ""
}

// Annotations returns the bundle path and user defined annotations from the
// libcontainer state.  We need to remove the bundle because that is a label
// added by libcontainer.
func Annotations(labels []string) (bundle string, userAnnotations map[string]string) {
	userAnnotations = make(map[string]string)
	for _, l := range labels {
		parts := strings.SplitN(l, "=", 2)
		if len(parts) < 2 {
			continue
		}
		if parts[0] == "bundle" {
			bundle = parts[1]
		} else {
			userAnnotations[parts[0]] = parts[1]
		}
	}
	return
}
