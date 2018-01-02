## nsenter

The `nsenter` package registers a special init constructor that is called before 
the Go runtime has a chance to boot.  This provides us the ability to `setns` on 
existing namespaces and avoid the issues that the Go runtime has with multiple 
threads.  This constructor will be called if this package is registered, 
imported, in your go application.

The `nsenter` package will `import "C"` and it uses [cgo](https://golang.org/cmd/cgo/)
package. In cgo, if the import of "C" is immediately preceded by a comment, that comment, 
called the preamble, is used as a header when compiling the C parts of the package.
So every time we  import package `nsenter`, the C code function `nsexec()` would be 
called. And package `nsenter` is now only imported in `main_unix.go`, so every time
before we call `cmd.Start` on linux, that C code would run.

Because `nsexec()` must be run before the Go runtime in order to use the
Linux kernel namespace, you must `import` this library into a package if
you plan to use `libcontainer` directly. Otherwise Go will not execute
the `nsexec()` constructor, which means that the re-exec will not cause
the namespaces to be joined. You can import it like this:

```go
import _ "github.com/opencontainers/runc/libcontainer/nsenter"
```

`nsexec()` will first get the file descriptor number for the init pipe
from the environment variable `_LIBCONTAINER_INITPIPE` (which was opened
by the parent and kept open across the fork-exec of the `nsexec()` init
process). The init pipe is used to read bootstrap data (namespace paths,
clone flags, uid and gid mappings, and the console path) from the parent
process. `nsexec()` will then call `setns(2)` to join the namespaces
provided in the bootstrap data (if available), `clone(2)` a child process
with the provided clone flags, update the user and group ID mappings, do
some further miscellaneous setup steps, and then send the PID of the
child process to the parent of the `nsexec()` "caller". Finally,
the parent `nsexec()` will exit and the child `nsexec()` process will
return to allow the Go runtime take over.

NOTE: We do both `setns(2)` and `clone(2)` even if we don't have any
CLONE_NEW* clone flags because we must fork a new process in order to
enter the PID namespace.



