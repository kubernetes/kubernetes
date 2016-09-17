go-syslog
=========

This repository provides a very simple `gsyslog` package. The point of this
package is to allow safe importing of syslog without introducing cross-compilation
issues. The stdlib `log/syslog` cannot be imported on Windows systems, and without
conditional compilation this adds complications.

Instead, `gsyslog` provides a very simple wrapper around `log/syslog` but returns
a runtime error if attempting to initialize on a non Linux or OSX system.

