# graceful-fs

graceful-fs functions as a drop-in replacement for the fs module,
making various improvements.

The improvements are meant to normalize behavior across different
platforms and environments, and to make filesystem access more
resilient to errors.

## Improvements over fs module

graceful-fs:

* keeps track of how many file descriptors are open, and by default
  limits this to 1024. Any further requests to open a file are put in a
  queue until new slots become available. If 1024 turns out to be too
  much, it decreases the limit further.
* fixes `lchmod` for Node versions prior to 0.6.2.
* implements `fs.lutimes` if possible. Otherwise it becomes a noop.
* ignores `EINVAL` and `EPERM` errors in `chown`, `fchown` or
  `lchown` if the user isn't root.
* makes `lchmod` and `lchown` become noops, if not available.
* retries reading a file if `read` results in EAGAIN error.

On Windows, it retries renaming a file for up to one second if `EACCESS`
or `EPERM` error occurs, likely because antivirus software has locked
the directory.

## Configuration

The maximum number of open file descriptors that graceful-fs manages may
be adjusted by setting `fs.MAX_OPEN` to a different number. The default
is 1024.
