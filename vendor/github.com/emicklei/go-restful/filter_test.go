package restful

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func setupServices(addGlobalFilter bool, addServiceFilter bool, addRouteFilter bool) {
	if addGlobalFilter {
		Filter(globalFilter)
	}
	Add(newTestService(addServiceFilter, addRouteFilter))
}

func tearDown() {
	DefaultContainer.webServices = []*WebService{}
	DefaultContainer.isRegisteredOnRoot = true // this allows for setupServices multiple times
	DefaultContainer.containerFilters = []FilterFunction{}
}

func newTestService(addServiceFilter bool, addRouteFilter bool) *WebService {
	ws := new(WebService).Path("")
	if addServiceFilter {
		ws.Filter(serviceFilter)
	}
	rb := ws.GET("/foo").To(foo)
	if addRouteFilter {
		rb.Filter(routeFilter)
	}
	ws.Route(rb)
	ws.Route(ws.GET("/bar").To(bar))
	return ws
}

func foo(req *Request, resp *Response) {
	io.WriteString(resp.ResponseWriter, "foo")
}

func bar(req *Request, resp *Response) {
	io.WriteString(resp.ResponseWriter, "bar")
}

func fail(req *Request, resp *Response) {
	http.Error(resp.ResponseWriter, "something failed", http.StatusInternalServerError)
}

func globalFilter(req *Request, resp *Response, chain *FilterChain) {
	io.WriteString(resp.ResponseWriter, "global-")
	chain.ProcessFilter(req, resp)
}

func serviceFilter(req *Request, resp *Response, chain *FilterChain) {
	io.WriteString(resp.ResponseWriter, "service-")
	chain.ProcessFilter(req, resp)
}

func routeFilter(req *Request, resp *Response, chain *FilterChain) {
	io.WriteString(resp.ResponseWriter, "route-")
	chain.ProcessFilter(req, resp)
}

func TestNoFilter(t *testing.T) {
	tearDown()
	setupServices(false, false, false)
	actual := sendIt("http://example.com/foo")
	if "foo" != actual {
		t.Fatal("expected: foo but got:" + actual)
	}
}

func TestGlobalFilter(t *testing.T) {
	tearDown()
	setupServices(true, false, false)
	actual := sendIt("http://example.com/foo")
	if "global-foo" != actual {
		t.Fatal("expected: global-foo but got:" + actual)
	}
}

func TestWebServiceFilter(t *testing.T) {
	tearDown()
	setupServices(true, true, false)
	actual := sendIt("http://example.com/foo")
	if "global-service-foo" != actual {
		t.Fatal("expected: global-service-foo but got:" + actual)
	}
}

func TestRouteFilter(t *testing.T) {
	tearDown()
	setupServices(true, true, true)
	actual := sendIt("http://example.com/foo")
	if "global-service-route-foo" != actual {
		t.Fatal("expected: global-service-route-foo but got:" + actual)
	}
}

func TestRouteFilterOnly(t *testing.T) {
	tearDown()
	setupServices(false, false, true)
	actual := sendIt("http://example.com/foo")
	if "route-foo" != actual {
		t.Fatal("expected: route-foo but got:" + actual)
	}
}

func TestBar(t *testing.T) {
	tearDown()
	setupServices(false, true, false)
	actual := sendIt("http://example.com/bar")
	if "service-bar" != actual {
		t.Fatal("expected: service-bar but got:" + actual)
	}
}

func TestAllFiltersBar(t *testing.T) {
	tearDown()
	setupServices(true, true, true)
	actual := sendIt("http://example.com/bar")
	if "global-service-bar" != actual {
		t.Fatal("expected: global-service-bar but got:" + actual)
	}
}

func sendIt(address string) string {
	httpRequest, _ := http.NewRequest("GET", address, nil)
	httpRequest.Header.Set("Accept", "*/*")
	httpWriter := httptest.NewRecorder()
	DefaultContainer.dispatch(httpWriter, httpRequest)
	return httpWriter.Body.String()
}

func sendItTo(address string, container *Container) string {
	httpRequest, _ := http.NewRequest("GET", address, nil)
	httpRequest.Header.Set("Accept", "*/*")
	httpWriter := httptest.NewRecorder()
	container.dispatch(httpWriter, httpRequest)
	return httpWriter.Body.String()
}
