// Copyright (c) 2017, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

// Package lint defines common interfaces for Go code checkers.
package lint // import "mvdan.cc/lint"

import (
	"go/token"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
)

// A Checker points out issues in a program.
type Checker interface {
	Program(*loader.Program)
	Check() ([]Issue, error)
}

type WithSSA interface {
	ProgramSSA(*ssa.Program)
}

// Issue represents an issue somewhere in a source code file.
type Issue interface {
	Pos() token.Pos
	Message() string
}
