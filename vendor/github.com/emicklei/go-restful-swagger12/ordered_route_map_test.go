package swagger

import (
	"testing"

	"github.com/emicklei/go-restful"
)

// go test -v -test.run TestOrderedRouteMap ...swagger
func TestOrderedRouteMap(t *testing.T) {
	m := newOrderedRouteMap()
	r1 := restful.Route{Path: "/r1"}
	r2 := restful.Route{Path: "/r2"}
	m.Add("a", r1)
	m.Add("b", r2)
	m.Add("b", r1)
	m.Add("d", r2)
	m.Add("c", r2)
	order := ""
	m.Do(func(k string, routes []restful.Route) {
		order += k
		if len(routes) == 0 {
			t.Fail()
		}
	})
	if order != "abdc" {
		t.Fail()
	}
}
