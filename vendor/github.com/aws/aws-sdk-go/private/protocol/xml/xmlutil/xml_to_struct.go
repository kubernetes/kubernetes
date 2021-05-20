package xmlutil

import (
	"encoding/xml"
	"fmt"
	"io"
	"sort"
)

// A XMLNode contains the values to be encoded or decoded.
type XMLNode struct {
	Name     xml.Name              `json:",omitempty"`
	Children map[string][]*XMLNode `json:",omitempty"`
	Text     string                `json:",omitempty"`
	Attr     []xml.Attr            `json:",omitempty"`

	namespaces map[string]string
	parent     *XMLNode
}

// textEncoder is a string type alias that implemnts the TextMarshaler interface.
// This alias type is used to ensure that the line feed (\n) (U+000A) is escaped.
type textEncoder string

func (t textEncoder) MarshalText() ([]byte, error) {
	return []byte(t), nil
}

// NewXMLElement returns a pointer to a new XMLNode initialized to default values.
func NewXMLElement(name xml.Name) *XMLNode {
	return &XMLNode{
		Name:     name,
		Children: map[string][]*XMLNode{},
		Attr:     []xml.Attr{},
	}
}

// AddChild adds child to the XMLNode.
func (n *XMLNode) AddChild(child *XMLNode) {
	child.parent = n
	if _, ok := n.Children[child.Name.Local]; !ok {
		n.Children[child.Name.Local] = []*XMLNode{}
	}
	n.Children[child.Name.Local] = append(n.Children[child.Name.Local], child)
}

// XMLToStruct converts a xml.Decoder stream to XMLNode with nested values.
func XMLToStruct(d *xml.Decoder, s *xml.StartElement) (*XMLNode, error) {
	out := &XMLNode{}
	for {
		tok, err := d.Token()
		if err != nil {
			if err == io.EOF {
				break
			} else {
				return out, err
			}
		}

		if tok == nil {
			break
		}

		switch typed := tok.(type) {
		case xml.CharData:
			out.Text = string(typed.Copy())
		case xml.StartElement:
			el := typed.Copy()
			out.Attr = el.Attr
			if out.Children == nil {
				out.Children = map[string][]*XMLNode{}
			}

			name := typed.Name.Local
			slice := out.Children[name]
			if slice == nil {
				slice = []*XMLNode{}
			}
			node, e := XMLToStruct(d, &el)
			out.findNamespaces()
			if e != nil {
				return out, e
			}
			node.Name = typed.Name
			node.findNamespaces()
			tempOut := *out
			// Save into a temp variable, simply because out gets squashed during
			// loop iterations
			node.parent = &tempOut
			slice = append(slice, node)
			out.Children[name] = slice
		case xml.EndElement:
			if s != nil && s.Name.Local == typed.Name.Local { // matching end token
				return out, nil
			}
			out = &XMLNode{}
		}
	}
	return out, nil
}

func (n *XMLNode) findNamespaces() {
	ns := map[string]string{}
	for _, a := range n.Attr {
		if a.Name.Space == "xmlns" {
			ns[a.Value] = a.Name.Local
		}
	}

	n.namespaces = ns
}

func (n *XMLNode) findElem(name string) (string, bool) {
	for node := n; node != nil; node = node.parent {
		for _, a := range node.Attr {
			namespace := a.Name.Space
			if v, ok := node.namespaces[namespace]; ok {
				namespace = v
			}
			if name == fmt.Sprintf("%s:%s", namespace, a.Name.Local) {
				return a.Value, true
			}
		}
	}
	return "", false
}

// StructToXML writes an XMLNode to a xml.Encoder as tokens.
func StructToXML(e *xml.Encoder, node *XMLNode, sorted bool) error {
	// Sort Attributes
	attrs := node.Attr
	if sorted {
		sortedAttrs := make([]xml.Attr, len(attrs))
		for _, k := range node.Attr {
			sortedAttrs = append(sortedAttrs, k)
		}
		sort.Sort(xmlAttrSlice(sortedAttrs))
		attrs = sortedAttrs
	}

	startElement := xml.StartElement{Name: node.Name, Attr: attrs}

	if node.Text != "" {
		e.EncodeElement(textEncoder(node.Text), startElement)
		return e.Flush()
	}

	e.EncodeToken(startElement)

	if sorted {
		sortedNames := []string{}
		for k := range node.Children {
			sortedNames = append(sortedNames, k)
		}
		sort.Strings(sortedNames)

		for _, k := range sortedNames {
			for _, v := range node.Children[k] {
				StructToXML(e, v, sorted)
			}
		}
	} else {
		for _, c := range node.Children {
			for _, v := range c {
				StructToXML(e, v, sorted)
			}
		}
	}

	e.EncodeToken(startElement.End())

	return e.Flush()
}
