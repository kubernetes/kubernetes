/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package generator

import (
	"fmt"
	"io"
	"runtime"
	"text/template"
)

// SnippetWriter is an attempt to make the template library usable.
// Methods are chainable, and you don't have to check Error() until you're all
// done.
type SnippetWriter struct {
	w           io.Writer
	context     *Context
	left, right string
	funcMap     template.FuncMap
	err         error
}

// w is the destination; left and right are the delimiters; @ and $ are both
// reasonable choices.
//
// c is used to make a function for every naming system, to which you can pass
// a type and get the corresponding name.
func NewSnippetWriter(w io.Writer, c *Context, left, right string) *SnippetWriter {
	sw := &SnippetWriter{
		w:       w,
		context: c,
		left:    left,
		right:   right,
		funcMap: template.FuncMap{},
	}
	for name, namer := range c.Namers {
		sw.funcMap[name] = namer.Name
	}
	return sw
}

// Do parses format and runs args through it. You can have arbitrary logic in
// the format (see the text/template documentation), but consider running many
// short templaces, with ordinary go logic in between--this may be more
// readable. Do is chainable. Any error causes every other call to do to be
// ignored, and the error will be returned by Error(). So you can check it just
// once, at the end of your function.
func (s *SnippetWriter) Do(format string, args interface{}) *SnippetWriter {
	if s.err != nil {
		return s
	}
	// Name the template by source file:line so it can be found when
	// there's an error.
	_, file, line, _ := runtime.Caller(1)
	tmpl, err := template.
		New(fmt.Sprintf("%s:%d", file, line)).
		Delims(s.left, s.right).
		Funcs(s.funcMap).
		Parse(format)
	if err != nil {
		s.err = err
		return s
	}
	err = tmpl.Execute(s.w, args)
	if err != nil {
		s.err = err
	}
	return s
}

// Error returns any encountered error.
func (s *SnippetWriter) Error() error {
	return s.err
}
