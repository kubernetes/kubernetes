// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"golang.org/x/text/message/pipeline"
)

// TODO:
// - merge information into existing files
// - handle different file formats (PO, XLIFF)
// - handle features (gender, plural)
// - message rewriting

func init() {
	lang = cmdExtract.Flag.String("lang", "en-US", "comma-separated list of languages to process")
}

var cmdExtract = &Command{
	Run:       runExtract,
	UsageLine: "extract <package>*",
	Short:     "extracts strings to be translated from code",
}

func runExtract(cmd *Command, config *pipeline.Config, args []string) error {
	config.Packages = args
	state, err := pipeline.Extract(config)
	if err != nil {
		return wrap(err, "extract failed")
	}
	if err := state.Import(); err != nil {
		return wrap(err, "import failed")
	}
	if err := state.Merge(); err != nil {
		return wrap(err, "merge failed")
	}
	return wrap(state.Export(), "export failed")
}
