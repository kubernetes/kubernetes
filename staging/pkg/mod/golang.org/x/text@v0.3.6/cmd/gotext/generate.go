// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"golang.org/x/text/message/pipeline"
)

func init() {
	out = cmdGenerate.Flag.String("out", "", "output file to write to")
}

var cmdGenerate = &Command{
	Run:       runGenerate,
	UsageLine: "generate <package>",
	Short:     "generates code to insert translated messages",
}

func runGenerate(cmd *Command, config *pipeline.Config, args []string) error {
	config.Packages = args
	s, err := pipeline.Extract(config)
	if err != nil {
		return wrap(err, "extraction failed")
	}
	if err := s.Import(); err != nil {
		return wrap(err, "import failed")
	}
	return wrap(s.Generate(), "generation failed")
}
