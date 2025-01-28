/*
Copyright 2015 The Kubernetes Authors.

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
	w       io.Writer
	context *Context
	// Left & right delimiters. text/template defaults to "{{" and "}}"
	// which is totally unusable for go code based templates.
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
// short templates with ordinary go logic in between--this may be more
// readable. Do is chainable. Any error causes every other call to do to be
// ignored, and the error will be returned by Error(). So you can check it just
// once, at the end of your function.
//
// 'args' can be quite literally anything; read the text/template documentation
// for details. Maps and structs work particularly nicely. Conveniently, the
// types package is designed to have structs that are easily referencable from
// the template language.
//
// Example:
//
// sw := generator.NewSnippetWriter(outBuffer, context, "$", "$")
// sw.Do(`The public type name is: $.type|public$`, map[string]interface{}{"type": t})
// return sw.Error()
//
// Where:
//   - "$" starts a template directive
//   - "." references the entire thing passed as args
//   - "type" therefore sees a map and looks up the key "type"
//   - "|" means "pass the thing on the left to the thing on the right"
//   - "public" is the name of a naming system, so the SnippetWriter has given
//     the template a function called "public" that takes a *types.Type and
//     returns the naming system's name. E.g., if the type is "string" this might
//     return "String".
//   - the second "$" ends the template directive.
//
// The map is actually not necessary. The below does the same thing:
//
// sw.Do(`The public type name is: $.|public$`, t)
//
// You may or may not find it more readable to use the map with a descriptive
// key, but if you want to pass more than one arg, the map or a custom struct
// becomes a requirement. You can do arbitrary logic inside these templates,
// but you should consider doing the logic in go and stitching them together
// for the sake of your readers.
//
// TODO: Change Do() to optionally take a list of pairs of parameters (key, value)
// and have it construct a combined map with that and args.
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

// Args exists to make it convenient to construct arguments for
// SnippetWriter.Do.
type Args map[interface{}]interface{}

// With makes a copy of a and adds the given key, value pair. If key overlaps,
// the new value wins.
func (a Args) With(key, value interface{}) Args {
	result := Args{}
	for k, v := range a {
		result[k] = v
	}
	result[key] = value
	return result
}

// WithArgs makes a copy of a and adds the given arguments. If any keys
// overlap, the values from rhs win.
func (a Args) WithArgs(rhs Args) Args {
	result := Args{}
	for k, v := range a {
		result[k] = v
	}
	for k, v := range rhs {
		result[k] = v
	}
	return result
}

func (s *SnippetWriter) Out() io.Writer {
	return s.w
}

// Error returns any encountered error.
func (s *SnippetWriter) Error() error {
	return s.err
}

// Dup creates an exact duplicate SnippetWriter with a different io.Writer.
func (s *SnippetWriter) Dup(w io.Writer) *SnippetWriter {
	ret := *s
	ret.w = w
	return &ret
}

// Append adds the contents of the io.Reader to this SnippetWriter's buffer.
func (s *SnippetWriter) Append(r io.Reader) error {
	_, err := io.Copy(s.w, r)
	return err
}
