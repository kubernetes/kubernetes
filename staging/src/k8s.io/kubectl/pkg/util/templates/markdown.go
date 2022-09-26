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
	"io"
	"strings"

	"github.com/russross/blackfriday/v2"
)

const linebreak = "\n"

// ASCIIRenderer implements blackfriday.Renderer
var _ blackfriday.Renderer = &ASCIIRenderer{}

// ASCIIRenderer is a blackfriday.Renderer intended for rendering markdown
// documents as plain text, well suited for human reading on terminals.
type ASCIIRenderer struct {
	Indentation string
}

func (r *ASCIIRenderer) GetFlags() int { return 0 }

// render markdown to text
func (r *ASCIIRenderer) RenderNode(w io.Writer, node *blackfriday.Node, entering bool) blackfriday.WalkStatus {
	switch node.Type {
	case blackfriday.Text:
		raw := string(node.Literal)
		lines := strings.Split(raw, linebreak)
		for _, line := range lines {
			trimmed := strings.Trim(line, " \n\t")
			if len(trimmed) > 0 && trimmed[0] != '_' {
				w.Write([]byte(" "))
			}
			w.Write([]byte(trimmed))
		}
	case blackfriday.HorizontalRule, blackfriday.Hardbreak:
		w.Write([]byte(linebreak + "----------" + linebreak))
	case blackfriday.Softbreak:
		w.Write([]byte(linebreak))
	case blackfriday.Code:
		w.Write([]byte(linebreak))
		lines := []string{}
		for _, line := range strings.Split(string(node.Literal), linebreak) {
			indented := r.Indentation + line
			lines = append(lines, indented)
		}
		w.Write([]byte(strings.Join(lines, linebreak)))
	case blackfriday.Image:
		w.Write(node.LinkData.Destination)
	case blackfriday.Link:
		w.Write([]byte(" "))
		w.Write(node.LinkData.Destination)
	case blackfriday.List:
		w.Write([]byte(linebreak))
		w.Write(node.Literal)
	case blackfriday.Paragraph:
		w.Write(node.Literal)
		w.Write([]byte(linebreak))
	default:
		w.Write(node.Literal)
	}
	return blackfriday.GoToNext
}

// RenderHeader writes document preamble and TOC if requested.
func (r *ASCIIRenderer) RenderHeader(w io.Writer, ast *blackfriday.Node) {

}

// RenderFooter writes document footer.
func (r *ASCIIRenderer) RenderFooter(w io.Writer, ast *blackfriday.Node) {
	io.WriteString(w, "\n")
}
