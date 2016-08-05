Package symlink implements EvalSymlinksInScope which is an extension of filepath.EvalSymlinks,
as well as a Windows long-path aware version of filepath.EvalSymlinks
from the [Go standard library](https://golang.org/pkg/path/filepath).

The code from filepath.EvalSymlinks has been adapted in fs.go.
Please read the LICENSE.BSD file that governs fs.go and LICENSE.APACHE for fs_test.go.
