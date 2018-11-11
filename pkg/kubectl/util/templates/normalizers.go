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

package templates

import (
	"strings"

	"github.com/MakeNowJust/heredoc"
	"github.com/russross/blackfriday"
	"github.com/spf13/cobra"
)

const Indentation = `  `

// LongDesc normalizes a command's long description to follow the conventions.
func LongDesc(s string) string {
	if len(s) == 0 {
		return s
	}
	return normalizer{s}.heredoc().markdown().trim().string
}

// Examples normalizes a command's examples to follow the conventions.
func Examples(s string) string {
	if len(s) == 0 {
		return s
	}
	return normalizer{s}.trim().indent().string
}

// Normalize perform all required normalizations on a given command.
func Normalize(cmd *cobra.Command) *cobra.Command {
	if len(cmd.Long) > 0 {
		cmd.Long = LongDesc(cmd.Long)
	}
	if len(cmd.Example) > 0 {
		cmd.Example = Examples(cmd.Example)
	}
	return cmd
}

// NormalizeAll perform all required normalizations in the entire command tree.
func NormalizeAll(cmd *cobra.Command) *cobra.Command {
	if cmd.HasSubCommands() {
		for _, subCmd := range cmd.Commands() {
			NormalizeAll(subCmd)
		}
	}
	Normalize(cmd)
	return cmd
}

type normalizer struct {
	string
}

func (s normalizer) markdown() normalizer {
	bytes := []byte(s.string)
	formatted := blackfriday.Markdown(bytes, &ASCIIRenderer{Indentation: Indentation}, 0)
	s.string = string(formatted)
	return s
}

func (s normalizer) heredoc() normalizer {
	s.string = heredoc.Doc(s.string)
	return s
}

func (s normalizer) trim() normalizer {
	s.string = strings.TrimSpace(s.string)
	return s
}

func (s normalizer) indent() normalizer {
	indentedLines := []string{}
	for _, line := range strings.Split(s.string, "\n") {
		trimmed := strings.TrimSpace(line)
		indented := Indentation + trimmed
		indentedLines = append(indentedLines, indented)
	}
	s.string = strings.Join(indentedLines, "\n")
	return s
}
