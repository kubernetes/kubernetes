// Package unix re-exports Linux specific parts of golang.org/x/sys/unix.
//
// It avoids breaking compilation on other OS by providing stubs as follows:
//   - Invoking a function always returns an error.
//   - Errnos have distinct, non-zero values.
//   - Constants have distinct but meaningless values.
//   - Types use the same names for members, but may or may not follow the
//     Linux layout.
package unix

// Note: please don't add any custom API to this package. Use internal/sys instead.
