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
called. And package `nsenter` is now only imported in Docker execdriver, so every time 
before we call `execdriver.Exec()`, that C code would run.

`nsexec()` will first check the environment variable `_LIBCONTAINER_INITPID` 
which will give the process of the container that should be joined. Namespaces fd will 
be found from `/proc/[pid]/ns` and set by `setns` syscall.

And then get the pipe number from `_LIBCONTAINER_INITPIPE`, error message could
be transfered through it. If tty is added, `_LIBCONTAINER_CONSOLE_PATH` will 
have value and start a console for output.

Finally, `nsexec()` will clone a child process , exit the parent process and let 
the Go runtime take over.
