package swagger

import (
	"bytes"
	"sort"
	"testing"
)

func TestSortParameters(t *testing.T) {
	unsorted := []Parameter{
		Parameter{
			Name:      "form2",
			ParamType: "form",
		},
		Parameter{
			Name:      "header1",
			ParamType: "header",
		},
		Parameter{
			Name:      "path2",
			ParamType: "path",
		},
		Parameter{
			Name:      "body",
			ParamType: "body",
		},
		Parameter{
			Name:      "path1",
			ParamType: "path",
		},
		Parameter{
			Name:      "form1",
			ParamType: "form",
		},
		Parameter{
			Name:      "query2",
			ParamType: "query",
		},
		Parameter{
			Name:      "query1",
			ParamType: "query",
		},
	}
	sort.Sort(ParameterSorter(unsorted))
	var b bytes.Buffer
	for _, p := range unsorted {
		b.WriteString(p.Name + ".")
	}
	if "path1.path2.query1.query2.form1.form2.header1.body." != b.String() {
		t.Fatal("sorting has changed:" + b.String())
	}
}
