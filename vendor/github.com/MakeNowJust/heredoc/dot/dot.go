// Copyright (c) 2014-2017 TSUYUSATO Kitsune
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php

// Package heredoc_dot is the set of shortcuts for dot import.
//
// For example:
//     package main
//
//     import (
//     	"fmt"
//     	"runtime"
//     	. "github.com/MakeNowJust/heredoc/dot"
//     )
//
//     func main() {
//     	fmt.Printf(D(`
//     		GOROOT: %s
//     		GOARCH: %s
//     		GOOS  : %s
//     	`), runtime.GOROOT(), runtime.GOARCH, runtime.GOOS)
//     }
package heredoc_dot

import "github.com/MakeNowJust/heredoc"

// Shortcut heredoc.Doc
func D(raw string) string {
	return heredoc.Doc(raw)
}

// Shortcut heredoc.Docf
func Df(raw string, args ...interface{}) string {
	return heredoc.Docf(raw, args...)
}
