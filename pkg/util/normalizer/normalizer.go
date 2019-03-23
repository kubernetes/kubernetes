/*
Copyright 2016 The Kubernetes Authors.

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

/*
This file is copied from /pkg/kubectl/cmd/templates/normalizer.go
In a future PR we should remove the original copy and use
/pkg/util/normalizer everywhere.
*/

package normalizer

import (
	"strings"

	"github.com/MakeNowJust/heredoc"
	"github.com/russross/blackfriday"
)

const indentation = `  `

// LongDesc normalizes a command's long description to follow the conventions.
func LongDesc(s string) string {
	if len(s) == 0 {
		return s
	}
	return normalizer{s}.Heredoc().Markdown().Trim().string
}

// Examples normalizes a command's examples to follow the conventions.
func Examples(s string) string {
	if len(s) == 0 {
		return s
	}
	return normalizer{s}.Trim().Indent().string
}

type normalizer struct {
	string
}

func (s normalizer) Markdown() normalizer {
	bytes := []byte(s.string)
	formatted := blackfriday.Markdown(bytes, &ASCIIRenderer{Indentation: indentation}, 0)
	s.string = string(formatted)
	return s
}

func (s normalizer) Heredoc() normalizer {
	s.string = heredoc.Doc(s.string)
	return s
}

func (s normalizer) Trim() normalizer {
	s.string = strings.TrimSpace(s.string)
	return s
}

func (s normalizer) Indent() normalizer {
	indentedLines := []string{}
	for _, line := range strings.Split(s.string, "\n") {
		trimmed := strings.TrimSpace(line)
		indented := indentation + trimmed
		indentedLines = append(indentedLines, indented)
	}
	s.string = strings.Join(indentedLines, "\n")
	return s
}
