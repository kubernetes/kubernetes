// Package filepath implements Go's filepath package with explicit
// operating systems (and for some functions and explicit working
// directory).  This allows tools built for one OS to operate on paths
// targeting another OS.  For example, a Linux build can determine
// whether a path is absolute on Linux or on Windows.
package filepath
