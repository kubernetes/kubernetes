package xmlutil

import (
	"encoding/xml"
	"io"
	"sort"
)

// A XMLNode contains the values to be encoded or decoded.
type XMLNode struct {
	Name     xml.Name              `json:",omitempty"`
	Children map[string][]*XMLNode `json:",omitempty"`
	Text     string                `json:",omitempty"`
	Attr     []xml.Attr            `json:",omitempty"`
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
		if tok == nil || err == io.EOF {
			break
		}
		if err != nil {
			return out, err
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
			if e != nil {
				return out, e
			}
			node.Name = typed.Name
			slice = append(slice, node)
			out.Children[name] = slice
		case xml.EndElement:
			if s != nil && s.Name.Local == typed.Name.Local { // matching end token
				return out, nil
			}
		}
	}
	return out, nil
}

// StructToXML writes an XMLNode to a xml.Encoder as tokens.
func StructToXML(e *xml.Encoder, node *XMLNode, sorted bool) error {
	e.EncodeToken(xml.StartElement{Name: node.Name, Attr: node.Attr})

	if node.Text != "" {
		e.EncodeToken(xml.CharData([]byte(node.Text)))
	} else if sorted {
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

	e.EncodeToken(xml.EndElement{Name: node.Name})
	return e.Flush()
}
