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

package util

import (
	"bytes"
	"go/doc"
	"io"
	"strings"
	"text/template"
)

func wrap(indent string, s string) string {
	var buf bytes.Buffer
	doc.ToText(&buf, s, indent, indent+"  ", 80-len(indent))
	return buf.String()
}

// ExecuteTemplate executes templateText with data and output written to w.
func ExecuteTemplate(w io.Writer, templateText string, data interface{}) error {
	t := template.New("top")
	t.Funcs(template.FuncMap{
		"trim": strings.TrimSpace,
		"wrap": wrap,
	})
	template.Must(t.Parse(templateText))
	return t.Execute(w, data)
}

func ExecuteTemplateToString(templateText string, data interface{}) (string, error) {
	b := bytes.Buffer{}
	err := ExecuteTemplate(&b, templateText, data)
	return b.String(), err
}
