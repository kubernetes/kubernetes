package md2man

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/russross/blackfriday/v2"
)

func fmtListFlags(flags blackfriday.ListType) string {
	knownFlags := []struct {
		name string
		flag blackfriday.ListType
	}{
		{"ListTypeOrdered", blackfriday.ListTypeOrdered},
		{"ListTypeDefinition", blackfriday.ListTypeDefinition},
		{"ListTypeTerm", blackfriday.ListTypeTerm},
		{"ListItemContainsBlock", blackfriday.ListItemContainsBlock},
		{"ListItemBeginningOfList", blackfriday.ListItemBeginningOfList},
		{"ListItemEndOfList", blackfriday.ListItemEndOfList},
	}

	var f []string
	for _, kf := range knownFlags {
		if flags&kf.flag != 0 {
			f = append(f, kf.name)
			flags &^= kf.flag
		}
	}
	if flags != 0 {
		f = append(f, fmt.Sprintf("Unknown(%#x)", flags))
	}
	return strings.Join(f, "|")
}

type debugDecorator struct {
	blackfriday.Renderer
}

func depth(node *blackfriday.Node) int {
	d := 0
	for n := node.Parent; n != nil; n = n.Parent {
		d++
	}
	return d
}

func (d *debugDecorator) RenderNode(w io.Writer, node *blackfriday.Node, entering bool) blackfriday.WalkStatus {
	fmt.Fprintf(os.Stderr, "%s%s %v %v\n",
		strings.Repeat("  ", depth(node)),
		map[bool]string{true: "+", false: "-"}[entering],
		node,
		fmtListFlags(node.ListFlags))
	var b strings.Builder
	status := d.Renderer.RenderNode(io.MultiWriter(&b, w), node, entering)
	if b.Len() > 0 {
		fmt.Fprintf(os.Stderr, ">> %q\n", b.String())
	}
	return status
}
