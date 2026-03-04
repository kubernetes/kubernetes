// This file is included to the build if any of the buildtags below
// are defined. Refer to README notes for more details.

//+build easyjson_nounsafe appengine

package jlexer

// bytesToStr creates a string normally from []byte
//
// Note that this method is roughly 1.5x slower than using the 'unsafe' method.
func bytesToStr(data []byte) string {
	return string(data)
}
