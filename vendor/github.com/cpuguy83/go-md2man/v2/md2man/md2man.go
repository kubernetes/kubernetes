package md2man

import (
	"os"
	"strconv"

	"github.com/russross/blackfriday/v2"
)

// Render converts a markdown document into a roff formatted document.
func Render(doc []byte) []byte {
	renderer := NewRoffRenderer()
	var r blackfriday.Renderer = renderer
	if v, _ := strconv.ParseBool(os.Getenv("MD2MAN_DEBUG")); v {
		r = &debugDecorator{Renderer: r}
	}

	return blackfriday.Run(doc,
		[]blackfriday.Option{
			blackfriday.WithRenderer(r),
			blackfriday.WithExtensions(renderer.GetExtensions()),
		}...)
}
