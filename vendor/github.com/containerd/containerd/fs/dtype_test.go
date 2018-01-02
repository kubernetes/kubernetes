package fs

import (
	"testing"

	"github.com/containerd/containerd/testutil"
)

func TestRequiresRootNOP(t *testing.T) {

	// This is a dummy test case that exist to call
	// testutil.RequiresRoot() on non-linux platforms.  This is
	// needed because the Makfile root-coverage tests target
	// determines which packages contain root test by grepping for
	// testutil.RequiresRoot.  Within the fs package, the only test
	// that references this symbol is in dtype_linux_test.go, but
	// that file is only built on linux.  Since the Makefile is not
	// go build tag aware it sees this file and then tries to run
	// the following command on all platforms: "go test ...
	// github.com/containerd/containerd/fs -test.root".  On
	// non-linux platforms this fails because there are no tests in
	// the "fs" package that reference testutil.RequiresRoot.  To
	// fix this problem we'll add a reference to this symbol below.

	testutil.RequiresRoot(t)
}
