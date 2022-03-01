// Package mountinfo provides a set of functions to retrieve information about OS mounts.
//
// Currently it supports Linux. For historical reasons, there is also some support for FreeBSD and OpenBSD,
// and a shallow implementation for Windows, but in general this is Linux-only package, so
// the rest of the document only applies to Linux, unless explicitly specified otherwise.
//
// In Linux, information about mounts seen by the current process is available from
// /proc/self/mountinfo. Note that due to mount namespaces, different processes can
// see different mounts. A per-process mountinfo table is available from /proc/<PID>/mountinfo,
// where <PID> is a numerical process identifier.
//
// In general, /proc is not a very efficient interface, and mountinfo is not an exception.
// For example, there is no way to get information about a specific mount point (i.e. it
// is all-or-nothing). This package tries to hide the /proc ineffectiveness by using
// parse filters while reading mountinfo. A filter can skip some entries, or stop
// processing the rest of the file once the needed information is found.
//
// For mountinfo filters that accept path as an argument, the path must be absolute,
// having all symlinks resolved, and being cleaned (i.e. no extra slashes or dots).
// One way to achieve all of the above is to employ filepath.Abs followed by
// filepath.EvalSymlinks (the latter calls filepath.Clean on the result so
// there is no need to explicitly call filepath.Clean).
//
// NOTE that in many cases there is no need to consult mountinfo at all. Here are some
// of the cases where mountinfo should not be parsed:
//
// 1. Before performing a mount. Usually, this is not needed, but if required (say to
// prevent over-mounts), to check whether a directory is mounted, call os.Lstat
// on it and its parent directory, and compare their st.Sys().(*syscall.Stat_t).Dev
// fields -- if they differ, then the directory is the mount point. NOTE this does
// not work for bind mounts. Optionally, the filesystem type can also be checked
// by calling unix.Statfs and checking the Type field (i.e. filesystem type).
//
// 2. After performing a mount. If there is no error returned, the mount succeeded;
// checking the mount table for a new mount is redundant and expensive.
//
// 3. Before performing an unmount. It is more efficient to do an unmount and ignore
// a specific error (EINVAL) which tells the directory is not mounted.
//
// 4. After performing an unmount. If there is no error returned, the unmount succeeded.
//
// 5. To find the mount point root of a specific directory. You can perform os.Stat()
// on the directory and traverse up until the Dev field of a parent directory differs.
package mountinfo
