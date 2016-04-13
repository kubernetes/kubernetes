package restful

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// go test -v -test.run TestCORSFilter_Preflight ...restful
// http://www.html5rocks.com/en/tutorials/cors/#toc-handling-a-not-so-simple-request
func TestCORSFilter_Preflight(t *testing.T) {
	tearDown()
	ws := new(WebService)
	ws.Route(ws.PUT("/cors").To(dummy))
	Add(ws)

	cors := CrossOriginResourceSharing{
		ExposeHeaders:  []string{"X-Custom-Header"},
		AllowedHeaders: []string{"X-Custom-Header", "X-Additional-Header"},
		CookiesAllowed: true,
		Container:      DefaultContainer}
	Filter(cors.Filter)

	// Preflight
	httpRequest, _ := http.NewRequest("OPTIONS", "http://api.alice.com/cors", nil)
	httpRequest.Method = "OPTIONS"
	httpRequest.Header.Set(HEADER_Origin, "http://api.bob.com")
	httpRequest.Header.Set(HEADER_AccessControlRequestMethod, "PUT")
	httpRequest.Header.Set(HEADER_AccessControlRequestHeaders, "X-Custom-Header, X-Additional-Header")

	httpWriter := httptest.NewRecorder()
	DefaultContainer.dispatch(httpWriter, httpRequest)

	actual := httpWriter.Header().Get(HEADER_AccessControlAllowOrigin)
	if "http://api.bob.com" != actual {
		t.Fatal("expected: http://api.bob.com but got:" + actual)
	}
	actual = httpWriter.Header().Get(HEADER_AccessControlAllowMethods)
	if "PUT" != actual {
		t.Fatal("expected: PUT but got:" + actual)
	}
	actual = httpWriter.Header().Get(HEADER_AccessControlAllowHeaders)
	if "X-Custom-Header, X-Additional-Header" != actual {
		t.Fatal("expected: X-Custom-Header, X-Additional-Header but got:" + actual)
	}

	if !cors.isOriginAllowed("somewhere") {
		t.Fatal("origin expected to be allowed")
	}
	cors.AllowedDomains = []string{"overthere.com"}
	if cors.isOriginAllowed("somewhere") {
		t.Fatal("origin [somewhere] expected NOT to be allowed")
	}
	if !cors.isOriginAllowed("overthere.com") {
		t.Fatal("origin [overthere] expected to be allowed")
	}

}

// go test -v -test.run TestCORSFilter_Actual ...restful
// http://www.html5rocks.com/en/tutorials/cors/#toc-handling-a-not-so-simple-request
func TestCORSFilter_Actual(t *testing.T) {
	tearDown()
	ws := new(WebService)
	ws.Route(ws.PUT("/cors").To(dummy))
	Add(ws)

	cors := CrossOriginResourceSharing{
		ExposeHeaders:  []string{"X-Custom-Header"},
		AllowedHeaders: []string{"X-Custom-Header", "X-Additional-Header"},
		CookiesAllowed: true,
		Container:      DefaultContainer}
	Filter(cors.Filter)

	// Actual
	httpRequest, _ := http.NewRequest("PUT", "http://api.alice.com/cors", nil)
	httpRequest.Header.Set(HEADER_Origin, "http://api.bob.com")
	httpRequest.Header.Set("X-Custom-Header", "value")

	httpWriter := httptest.NewRecorder()
	DefaultContainer.dispatch(httpWriter, httpRequest)
	actual := httpWriter.Header().Get(HEADER_AccessControlAllowOrigin)
	if "http://api.bob.com" != actual {
		t.Fatal("expected: http://api.bob.com but got:" + actual)
	}
	if httpWriter.Body.String() != "dummy" {
		t.Fatal("expected: dummy but got:" + httpWriter.Body.String())
	}
}

var allowedDomainInput = []struct {
	domains  []string
	origin   string
	accepted bool
}{
	{[]string{}, "http://anything.com", true},
}

// go test -v -test.run TestCORSFilter_AllowedDomains ...restful
func TestCORSFilter_AllowedDomains(t *testing.T) {
	for _, each := range allowedDomainInput {
		tearDown()
		ws := new(WebService)
		ws.Route(ws.PUT("/cors").To(dummy))
		Add(ws)

		cors := CrossOriginResourceSharing{
			AllowedDomains: each.domains,
			CookiesAllowed: true,
			Container:      DefaultContainer}
		Filter(cors.Filter)

		httpRequest, _ := http.NewRequest("PUT", "http://api.his.com/cors", nil)
		httpRequest.Header.Set(HEADER_Origin, each.origin)
		httpWriter := httptest.NewRecorder()
		DefaultContainer.dispatch(httpWriter, httpRequest)
		actual := httpWriter.Header().Get(HEADER_AccessControlAllowOrigin)
		if actual != each.origin && each.accepted {
			t.Fatal("expected to be accepted")
		}
		if actual == each.origin && !each.accepted {
			t.Fatal("did not expect to be accepted")
		}
	}
}
