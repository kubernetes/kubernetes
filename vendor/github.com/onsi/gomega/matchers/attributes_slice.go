package matchers

import (
	"encoding/xml"
	"strings"
)

type attributesSlice []xml.Attr

func (attrs attributesSlice) Len() int { return len(attrs) }
func (attrs attributesSlice) Less(i, j int) bool {
	if attrs[i].Name.Local != attrs[j].Name.Local {
		return strings.Compare(attrs[i].Name.Local, attrs[j].Name.Local) == -1
	}
	if attrs[i].Name.Space != attrs[j].Name.Space {
		return strings.Compare(attrs[i].Name.Space, attrs[j].Name.Space) == -1
	}
	return strings.Compare(attrs[i].Value, attrs[j].Value) == -1
}
func (attrs attributesSlice) Swap(i, j int) { attrs[i], attrs[j] = attrs[j], attrs[i] }
