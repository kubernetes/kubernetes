/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"text/template"
)

// gotemplate is a simple CLI for text/template. It reads from stdin and writes to stdout.
// Optional arguments are <key>=<value> pairs which can be used as {{.<key>}} to inject
// the <value> for that key.
//
// Besides the default functions (https://pkg.go.dev/text/template#hdr-Functions),
// gotemplate also implements:
// - include <filename>: returns the content of that file as string
// - indent <number of spaces> <string>: replace each newline with "newline + spaces", indent the newline at the end
// - trim <string>: strip leading and trailing whitespace

func main() {
	kvs := make(map[string]string)

	for _, keyValue := range os.Args[1:] {
		index := strings.Index(keyValue, "=")
		if index <= 0 {
			fmt.Fprintf(os.Stderr, "optional arguments must be of the form <key>=<value>, got instead: %q\n", keyValue)
			os.Exit(1)
		}
		kvs[keyValue[0:index]] = keyValue[index+1:]
	}

	if err := generate(os.Stdin, os.Stdout, kvs); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func generate(in io.Reader, out io.Writer, data interface{}) error {
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(in); err != nil {
		return fmt.Errorf("reading input: %v", err)
	}

	funcMap := template.FuncMap{
		"include": include,
		"indent":  indent,
		"trim":    trim,
	}

	tmpl, err := template.New("").Funcs(funcMap).Parse(buf.String())
	if err != nil {
		return fmt.Errorf("parsing input as text template: %v", err)
	}

	if err := tmpl.Execute(out, data); err != nil {
		return fmt.Errorf("generating result: %v", err)
	}
	return nil
}

func include(filename string) (string, error) {
	content, err := os.ReadFile(filename)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

func indent(numSpaces int, content string) string {
	if content == "" {
		return ""
	}
	prefix := strings.Repeat(" ", numSpaces)
	return strings.ReplaceAll(content, "\n", "\n"+prefix)
}

func trim(content string) string {
	return strings.TrimSpace(content)
}
