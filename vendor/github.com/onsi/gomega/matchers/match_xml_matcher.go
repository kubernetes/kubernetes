package matchers

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"

	"github.com/onsi/gomega/format"
	"golang.org/x/net/html/charset"
)

type MatchXMLMatcher struct {
	XMLToMatch interface{}
}

func (matcher *MatchXMLMatcher) Match(actual interface{}) (success bool, err error) {
	actualString, expectedString, err := matcher.formattedPrint(actual)
	if err != nil {
		return false, err
	}

	aval, err := parseXmlContent(actualString)
	if err != nil {
		return false, fmt.Errorf("Actual '%s' should be valid XML, but it is not.\nUnderlying error:%s", actualString, err)
	}

	eval, err := parseXmlContent(expectedString)
	if err != nil {
		return false, fmt.Errorf("Expected '%s' should be valid XML, but it is not.\nUnderlying error:%s", expectedString, err)
	}

	return reflect.DeepEqual(aval, eval), nil
}

func (matcher *MatchXMLMatcher) FailureMessage(actual interface{}) (message string) {
	actualString, expectedString, _ := matcher.formattedPrint(actual)
	return fmt.Sprintf("Expected\n%s\nto match XML of\n%s", actualString, expectedString)
}

func (matcher *MatchXMLMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	actualString, expectedString, _ := matcher.formattedPrint(actual)
	return fmt.Sprintf("Expected\n%s\nnot to match XML of\n%s", actualString, expectedString)
}

func (matcher *MatchXMLMatcher) formattedPrint(actual interface{}) (actualString, expectedString string, err error) {
	var ok bool
	actualString, ok = toString(actual)
	if !ok {
		return "", "", fmt.Errorf("MatchXMLMatcher matcher requires a string, stringer, or []byte.  Got actual:\n%s", format.Object(actual, 1))
	}
	expectedString, ok = toString(matcher.XMLToMatch)
	if !ok {
		return "", "", fmt.Errorf("MatchXMLMatcher matcher requires a string, stringer, or []byte.  Got expected:\n%s", format.Object(matcher.XMLToMatch, 1))
	}
	return actualString, expectedString, nil
}

func parseXmlContent(content string) (*xmlNode, error) {
	allNodes := []*xmlNode{}

	dec := newXmlDecoder(strings.NewReader(content))
	for {
		tok, err := dec.Token()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("failed to decode next token: %v", err) // untested section
		}

		lastNodeIndex := len(allNodes) - 1
		var lastNode *xmlNode
		if len(allNodes) > 0 {
			lastNode = allNodes[lastNodeIndex]
		} else {
			lastNode = &xmlNode{}
		}

		switch tok := tok.(type) {
		case xml.StartElement:
			attrs := attributesSlice(tok.Attr)
			sort.Sort(attrs)
			allNodes = append(allNodes, &xmlNode{XMLName: tok.Name, XMLAttr: tok.Attr})
		case xml.EndElement:
			if len(allNodes) > 1 {
				allNodes[lastNodeIndex-1].Nodes = append(allNodes[lastNodeIndex-1].Nodes, lastNode)
				allNodes = allNodes[:lastNodeIndex]
			}
		case xml.CharData:
			lastNode.Content = append(lastNode.Content, tok.Copy()...)
		case xml.Comment:
			lastNode.Comments = append(lastNode.Comments, tok.Copy()) // untested section
		case xml.ProcInst:
			lastNode.ProcInsts = append(lastNode.ProcInsts, tok.Copy())
		}
	}

	if len(allNodes) == 0 {
		return nil, errors.New("found no nodes")
	}
	firstNode := allNodes[0]
	trimParentNodesContentSpaces(firstNode)

	return firstNode, nil
}

func newXmlDecoder(reader io.Reader) *xml.Decoder {
	dec := xml.NewDecoder(reader)
	dec.CharsetReader = charset.NewReaderLabel
	return dec
}

func trimParentNodesContentSpaces(node *xmlNode) {
	if len(node.Nodes) > 0 {
		node.Content = bytes.TrimSpace(node.Content)
		for _, childNode := range node.Nodes {
			trimParentNodesContentSpaces(childNode)
		}
	}
}

type xmlNode struct {
	XMLName   xml.Name
	Comments  []xml.Comment
	ProcInsts []xml.ProcInst
	XMLAttr   []xml.Attr
	Content   []byte
	Nodes     []*xmlNode
}
