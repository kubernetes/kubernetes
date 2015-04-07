package restful

import (
	"net/http"
	"testing"
)

// go test -v -test.run TestContainer_computeAllowedMethods ...restful
func TestContainer_computeAllowedMethods(t *testing.T) {
	wc := NewContainer()
	ws1 := new(WebService).Path("/users")
	ws1.Route(ws1.GET("{i}").To(dummy))
	ws1.Route(ws1.POST("{i}").To(dummy))
	wc.Add(ws1)
	httpRequest, _ := http.NewRequest("GET", "http://api.his.com/users/1", nil)
	rreq := Request{Request: httpRequest}
	m := wc.computeAllowedMethods(&rreq)
	if len(m) != 2 {
		t.Errorf("got %d expected 2 methods, %v", len(m), m)
	}
}
