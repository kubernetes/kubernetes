package matchers

import (
	"encoding/xml"
	"strings"
)

type attributesSlice []xml.Attr

func (attrs attributesSlice) Len() int { return len(attrs) }
func (attrs attributesSlice) Less(i, j int) bool {
	return strings.Compare(attrs[i].Name.Local, attrs[j].Name.Local) == -1
}
func (attrs attributesSlice) Swap(i, j int) { attrs[i], attrs[j] = attrs[j], attrs[i] }
