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

package generator_test

import (
	"bytes"
	"strings"
	"testing"

	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/parser"
)

func construct(t *testing.T, files map[string]string) *generator.Context {
	b := parser.New()
	for name, src := range files {
		if err := b.AddFile("/tmp/"+name, name, []byte(src)); err != nil {
			t.Fatal(err)
		}
	}
	c, err := generator.NewContext(b, namer.NameSystems{
		"public":  namer.NewPublicNamer(0),
		"private": namer.NewPrivateNamer(0),
	}, "public")
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestSnippetWriter(t *testing.T) {
	var structTest = map[string]string{
		"base/foo/proto/foo.go": `
package foo

// Blah is a test.
// A test, I tell you.
type Blah struct {
	// A is the first field.
	A int64 ` + "`" + `json:"a"` + "`" + `

	// B is the second field.
	// Multiline comments work.
	B string ` + "`" + `json:"b"` + "`" + `
}
`,
	}

	c := construct(t, structTest)
	b := &bytes.Buffer{}
	err := generator.NewSnippetWriter(b, c, "$", "$").
		Do("$.|public$$.|private$", c.Order[0]).
		Error()
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := "Blahblah", b.String(); e != a {
		t.Errorf("Expected %q, got %q", e, a)
	}

	err = generator.NewSnippetWriter(b, c, "$", "$").
		Do("$.|public", c.Order[0]).
		Error()
	if err == nil {
		t.Errorf("expected error on invalid template")
	} else {
		// Dear reader, I apologize for making the worst change
		// detection test in the history of ever.
		if e, a := "snippet_writer_test.go:78", err.Error(); !strings.Contains(a, e) {
			t.Errorf("Expected %q but didn't find it in %q", e, a)
		}
	}
}
