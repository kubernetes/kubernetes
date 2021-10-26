package builder3

import (
	"sort"

	"github.com/emicklei/go-restful"
	"k8s.io/kube-openapi/pkg/spec3"
)

func mapKeyFromParam(param *restful.Parameter) interface{} {
	return struct {
		Name string
		Kind int
	}{
		Name: param.Data().Name,
		Kind: param.Data().Kind,
	}
}

func (s parameters) Len() int      { return len(s) }
func (s parameters) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

type parameters []*spec3.Parameter

type byNameIn struct {
	parameters
}

func (s byNameIn) Less(i, j int) bool {
	return s.parameters[i].Name < s.parameters[j].Name || (s.parameters[i].Name == s.parameters[j].Name && s.parameters[i].In < s.parameters[j].In)
}

// SortParameters sorts parameters by Name and In fields.
func sortParameters(p []*spec3.Parameter) {
	sort.Sort(byNameIn{p})
}
