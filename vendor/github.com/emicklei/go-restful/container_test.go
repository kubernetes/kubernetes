package restful

import (
	"net/http"
	"net/http/httptest"
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

func TestContainer_HandleWithFilter(t *testing.T) {
	prefilterCalled := false
	postfilterCalled := false
	httpHandlerCalled := false

	wc := NewContainer()
	wc.Filter(func(request *Request, response *Response, chain *FilterChain) {
		prefilterCalled = true
		chain.ProcessFilter(request, response)
	})
	wc.HandleWithFilter("/", http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		httpHandlerCalled = true
		w.Write([]byte("ok"))
	}))
	wc.Filter(func(request *Request, response *Response, chain *FilterChain) {
		postfilterCalled = true
		chain.ProcessFilter(request, response)
	})

	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/", nil)
	wc.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("unexpected code %d", recorder.Code)
	}
	if recorder.Body.String() != "ok" {
		t.Errorf("unexpected body %s", recorder.Body.String())
	}
	if !prefilterCalled {
		t.Errorf("filter added before calling HandleWithFilter wasn't called")
	}
	if !postfilterCalled {
		t.Errorf("filter added after calling HandleWithFilter wasn't called")
	}
	if !httpHandlerCalled {
		t.Errorf("handler added by calling HandleWithFilter wasn't called")
	}
}

func TestContainerAddAndRemove(t *testing.T) {
	ws1 := new(WebService).Path("/")
	ws2 := new(WebService).Path("/users")
	wc := NewContainer()
	wc.Add(ws1)
	wc.Add(ws2)
	wc.Remove(ws2)
	if len(wc.webServices) != 1 {
		t.Errorf("expected one webservices")
	}
	if !wc.isRegisteredOnRoot {
		t.Errorf("expected on root registered")
	}
	wc.Remove(ws1)
	if len(wc.webServices) > 0 {
		t.Errorf("expected zero webservices")
	}
	if wc.isRegisteredOnRoot {
		t.Errorf("expected not on root registered")
	}
}
