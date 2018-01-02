// Copyright 2016 The Linux Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package schema_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/url"
	"os"
	"strings"
	"testing"

	"github.com/opencontainers/image-spec/schema"
	"github.com/pkg/errors"
	"github.com/russross/blackfriday"
)

var (
	errFormatInvalid = errors.New("format: invalid")
)

func TestValidateDescriptor(t *testing.T) {
	validate(t, "../descriptor.md")
}

func TestValidateManifest(t *testing.T) {
	validate(t, "../manifest.md")
}

func TestValidateImageIndex(t *testing.T) {
	validate(t, "../image-index.md")
}

func TestValidateImageLayout(t *testing.T) {
	validate(t, "../image-layout.md")
}

func TestValidateConfig(t *testing.T) {
	validate(t, "../config.md")
}

// TODO(sur): include examples from all specification files
func validate(t *testing.T, name string) {
	m, err := os.Open(name)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	examples, err := extractExamples(m)
	if err != nil {
		t.Fatal(err)
	}

	for _, example := range examples {
		if example.Err == errFormatInvalid && example.Mediatype == "" { // ignore
			continue
		}

		if example.Err != nil {
			printFields(t, "error", example.Mediatype, example.Title, example.Err)
			t.Error(err)
			continue
		}

		err = schema.Validator(example.Mediatype).Validate(strings.NewReader(example.Body))
		if err == nil {
			printFields(t, "ok", example.Mediatype, example.Title)
			t.Log(example.Body, "---")
			continue
		}

		var errs []error
		if verr, ok := errors.Cause(err).(schema.ValidationError); ok {
			errs = verr.Errs
		} else {
			printFields(t, "error", example.Mediatype, example.Title, err)
			t.Error(err)
			t.Log(example.Body, "---")
			continue
		}

		for _, err := range errs {
			// TOOD(stevvooe): This is nearly useless without file, line no.
			printFields(t, "invalid", example.Mediatype, example.Title)
			t.Error(err)
			fmt.Println(example.Body, "---")
			continue
		}
	}
}

// renderer allows one to incercept fenced blocks in markdown documents.
type renderer struct {
	blackfriday.Renderer
	fn func(text []byte, lang string)
}

func (r *renderer) BlockCode(out *bytes.Buffer, text []byte, lang string) {
	r.fn(text, lang)
	r.Renderer.BlockCode(out, text, lang)
}

type example struct {
	Lang      string // gets raw "lang" field
	Title     string
	Mediatype string
	Body      string
	Err       error

	// TODO(stevvooe): Figure out how to keep track of revision, file, line so
	// that we can trace back verification output.
}

// parseExample treats the field as a syntax,attribute tuple separated by a comma.
// Attributes are encoded as a url values.
//
// An example of this is `json,title=Foo%20Bar&mediatype=application/json. We
// get that the "lang" is json, the title is "Foo Bar" and the mediatype is
// "application/json".
//
// This preserves syntax highlighting and lets us tag examples with further
// metadata.
func parseExample(lang, body string) (e example) {
	e.Lang = lang
	e.Body = body

	parts := strings.SplitN(lang, ",", 2)
	if len(parts) < 2 {
		e.Err = errFormatInvalid
		return
	}

	m, err := url.ParseQuery(parts[1])
	if err != nil {
		e.Err = err
		return
	}

	e.Mediatype = m.Get("mediatype")
	e.Title = m.Get("title")
	return
}

func extractExamples(rd io.Reader) ([]example, error) {
	p, err := ioutil.ReadAll(rd)
	if err != nil {
		return nil, err
	}

	var examples []example
	renderer := &renderer{
		Renderer: blackfriday.HtmlRenderer(0, "test test", ""),
		fn: func(text []byte, lang string) {
			examples = append(examples, parseExample(lang, string(text)))
		},
	}

	// just pass over the markdown and ignore the rendered result. We just want
	// the side-effect of calling back for each code block.
	// TODO(stevvooe): Consider just parsing these with a scanner. It will be
	// faster and we can retain file, line no.
	blackfriday.MarkdownOptions(p, renderer, blackfriday.Options{
		Extensions: blackfriday.EXTENSION_FENCED_CODE,
	})

	return examples, nil
}

// printFields prints each value tab separated.
func printFields(t *testing.T, vs ...interface{}) {
	var ss []string
	for _, f := range vs {
		ss = append(ss, fmt.Sprint(f))
	}
	t.Log(strings.Join(ss, "\t"))
}
