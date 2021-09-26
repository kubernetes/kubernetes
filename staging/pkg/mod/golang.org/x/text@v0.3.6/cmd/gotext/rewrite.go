// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io"
	"os"

	"golang.org/x/text/message/pipeline"
)

const printerType = "golang.org/x/text/message.Printer"

// TODO:
// - merge information into existing files
// - handle different file formats (PO, XLIFF)
// - handle features (gender, plural)
// - message rewriting

func init() {
	overwrite = cmdRewrite.Flag.Bool("w", false, "write files in place")
}

var (
	overwrite *bool
)

var cmdRewrite = &Command{
	Run:       runRewrite,
	UsageLine: "rewrite <package>",
	Short:     "rewrites fmt functions to use a message Printer",
	Long: `
rewrite is typically done once for a project. It rewrites all usages of
fmt to use x/text's message package whenever a message.Printer is in scope.
It rewrites Print and Println calls with constant strings to the equivalent
using Printf to allow translators to reorder arguments.
`,
}

func runRewrite(cmd *Command, _ *pipeline.Config, args []string) error {
	var w io.Writer
	if !*overwrite {
		w = os.Stdout
	}
	pkg := "."
	switch len(args) {
	case 0:
	case 1:
		pkg = args[0]
	default:
		return errorf("can only specify at most one package")
	}
	return pipeline.Rewrite(w, pkg)
}
