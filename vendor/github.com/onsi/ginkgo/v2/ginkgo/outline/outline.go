package outline

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"strings"

	"golang.org/x/tools/go/ast/inspector"
)

const (
	// ginkgoImportPath is the well-known ginkgo import path
	ginkgoImportPath = "github.com/onsi/ginkgo/v2"
)

// FromASTFile returns an outline for a Ginkgo test source file
func FromASTFile(fset *token.FileSet, src *ast.File) (*outline, error) {
	ginkgoPackageName := packageNameForImport(src, ginkgoImportPath)
	if ginkgoPackageName == nil {
		return nil, fmt.Errorf("file does not import %q", ginkgoImportPath)
	}

	root := ginkgoNode{}
	stack := []*ginkgoNode{&root}
	ispr := inspector.New([]*ast.File{src})
	ispr.Nodes([]ast.Node{(*ast.CallExpr)(nil)}, func(node ast.Node, push bool) bool {
		if push {
			// Pre-order traversal
			ce, ok := node.(*ast.CallExpr)
			if !ok {
				// Because `Nodes` calls this function only when the node is an
				// ast.CallExpr, this should never happen
				panic(fmt.Errorf("node starting at %d, ending at %d is not an *ast.CallExpr", node.Pos(), node.End()))
			}
			gn, ok := ginkgoNodeFromCallExpr(fset, ce, ginkgoPackageName)
			if !ok {
				// Node is not a Ginkgo spec or container, continue
				return true
			}
			parent := stack[len(stack)-1]
			parent.Nodes = append(parent.Nodes, gn)
			stack = append(stack, gn)
			return true
		}
		// Post-order traversal
		start, end := absoluteOffsetsForNode(fset, node)
		lastVisitedGinkgoNode := stack[len(stack)-1]
		if start != lastVisitedGinkgoNode.Start || end != lastVisitedGinkgoNode.End {
			// Node is not a Ginkgo spec or container, so it was not pushed onto the stack, continue
			return true
		}
		stack = stack[0 : len(stack)-1]
		return true
	})
	if len(root.Nodes) == 0 {
		return &outline{[]*ginkgoNode{}}, nil
	}

	// Derive the final focused property for all nodes. This must be done
	// _before_ propagating the inherited focused property.
	root.BackpropagateUnfocus()
	// Now, propagate inherited properties, including focused and pending.
	root.PropagateInheritedProperties()

	return &outline{root.Nodes}, nil
}

type outline struct {
	Nodes []*ginkgoNode `json:"nodes"`
}

func (o *outline) MarshalJSON() ([]byte, error) {
	return json.Marshal(o.Nodes)
}

// String returns a CSV-formatted outline. Spec or container are output in
// depth-first order.
func (o *outline) String() string {
	return o.StringIndent(0)
}

// StringIndent returns a CSV-formated outline, but every line is indented by
// one 'width' of spaces for every level of nesting.
func (o *outline) StringIndent(width int) string {
	var b bytes.Buffer
	b.WriteString("Name,Text,Start,End,Spec,Focused,Pending,Labels\n")

	csvWriter := csv.NewWriter(&b)

	currentIndent := 0
	pre := func(n *ginkgoNode) {
		b.WriteString(fmt.Sprintf("%*s", currentIndent, ""))
		var labels string
		if len(n.Labels) == 1 {
			labels = n.Labels[0]
		} else {
			labels = strings.Join(n.Labels, ", ")
		}

		row := []string{
			n.Name,
			n.Text,
			strconv.Itoa(n.Start),
			strconv.Itoa(n.End),
			strconv.FormatBool(n.Spec),
			strconv.FormatBool(n.Focused),
			strconv.FormatBool(n.Pending),
			labels,
		}
		csvWriter.Write(row)

		// Ensure we write to `b' before the next `b.WriteString()', which might be adding indentation
		csvWriter.Flush()

		currentIndent += width
	}
	post := func(n *ginkgoNode) {
		currentIndent -= width
	}
	for _, n := range o.Nodes {
		n.Walk(pre, post)
	}

	return b.String()
}
