package facts

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"reflect"
	"strings"

	"golang.org/x/tools/go/analysis"
)

type Generator int

// A list of known generators we can detect
const (
	Unknown Generator = iota
	Goyacc
	Cgo
	Stringer
	ProtocGenGo
)

var (
	// used by cgo before Go 1.11
	oldCgo = []byte("// Created by cgo - DO NOT EDIT")
	prefix = []byte("// Code generated ")
	suffix = []byte(" DO NOT EDIT.")
	nl     = []byte("\n")
	crnl   = []byte("\r\n")
)

func isGenerated(path string) (Generator, bool) {
	f, err := os.Open(path)
	if err != nil {
		return 0, false
	}
	defer f.Close()
	br := bufio.NewReader(f)
	for {
		s, err := br.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return 0, false
		}
		s = bytes.TrimSuffix(s, crnl)
		s = bytes.TrimSuffix(s, nl)
		if bytes.HasPrefix(s, prefix) && bytes.HasSuffix(s, suffix) {
			text := string(s[len(prefix) : len(s)-len(suffix)])
			switch text {
			case "by goyacc.":
				return Goyacc, true
			case "by cmd/cgo;":
				return Cgo, true
			case "by protoc-gen-go.":
				return ProtocGenGo, true
			}
			if strings.HasPrefix(text, `by "stringer `) {
				return Stringer, true
			}
			if strings.HasPrefix(text, `by goyacc `) {
				return Goyacc, true
			}

			return Unknown, true
		}
		if bytes.Equal(s, oldCgo) {
			return Cgo, true
		}
		if err == io.EOF {
			break
		}
	}
	return 0, false
}

var Generated = &analysis.Analyzer{
	Name: "isgenerated",
	Doc:  "annotate file names that have been code generated",
	Run: func(pass *analysis.Pass) (interface{}, error) {
		m := map[string]Generator{}
		for _, f := range pass.Files {
			path := pass.Fset.PositionFor(f.Pos(), false).Filename
			g, ok := isGenerated(path)
			if ok {
				m[path] = g
			}
		}
		return m, nil
	},
	RunDespiteErrors: true,
	ResultType:       reflect.TypeOf(map[string]Generator{}),
}
