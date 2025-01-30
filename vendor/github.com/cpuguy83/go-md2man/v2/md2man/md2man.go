package md2man

import (
	"github.com/russross/blackfriday/v2"
)

// Render converts a markdown document into a roff formatted document.
func Render(doc []byte) []byte {
	renderer := NewRoffRenderer()

	return blackfriday.Run(doc,
		[]blackfriday.Option{
			blackfriday.WithRenderer(renderer),
			blackfriday.WithExtensions(renderer.GetExtensions()),
		}...)
}
