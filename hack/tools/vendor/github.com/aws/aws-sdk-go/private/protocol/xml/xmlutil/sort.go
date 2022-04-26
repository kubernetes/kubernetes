package xmlutil

import (
	"encoding/xml"
	"strings"
)

type xmlAttrSlice []xml.Attr

func (x xmlAttrSlice) Len() int {
	return len(x)
}

func (x xmlAttrSlice) Less(i, j int) bool {
	spaceI, spaceJ := x[i].Name.Space, x[j].Name.Space
	localI, localJ := x[i].Name.Local, x[j].Name.Local
	valueI, valueJ := x[i].Value, x[j].Value

	spaceCmp := strings.Compare(spaceI, spaceJ)
	localCmp := strings.Compare(localI, localJ)
	valueCmp := strings.Compare(valueI, valueJ)

	if spaceCmp == -1 || (spaceCmp == 0 && (localCmp == -1 || (localCmp == 0 && valueCmp == -1))) {
		return true
	}

	return false
}

func (x xmlAttrSlice) Swap(i, j int) {
	x[i], x[j] = x[j], x[i]
}
