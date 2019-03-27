package httptreemux

import (
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
)

func simpleHandler(w http.ResponseWriter, r *http.Request, params map[string]string) {}

func panicHandler(w http.ResponseWriter, r *http.Request, params map[string]string) {
	panic("test panic")
}

func newRequest(method, path string, body io.Reader) (*http.Request, error) {
	r, _ := http.NewRequest(method, path, body)
	u, _ := url.ParseRequestURI(path)
	r.URL = u
	r.RequestURI = path
	return r, nil
}

type RequestCreator func(string, string, io.Reader) (*http.Request, error)
type TestScenario struct {
	RequestCreator RequestCreator
	ServeStyle     bool
	description    string
}

var scenarios = []TestScenario{
	TestScenario{newRequest, false, "Test with RequestURI and normal ServeHTTP"},
	TestScenario{http.NewRequest, false, "Test with URL.Path and normal ServeHTTP"},
	TestScenario{newRequest, true, "Test with RequestURI and LookupResult"},
	TestScenario{http.NewRequest, true, "Test with URL.Path and LookupResult"},
}

// This type and the benchRequest function are modified from go-http-routing-benchmark.
type mockResponseWriter struct {
	code        int
	calledWrite bool
}

func (m *mockResponseWriter) Header() (h http.Header) {
	return http.Header{}
}

func (m *mockResponseWriter) Write(p []byte) (n int, err error) {
	m.calledWrite = true
	return len(p), nil
}

func (m *mockResponseWriter) WriteString(s string) (n int, err error) {
	m.calledWrite = true
	return len(s), nil
}

func (m *mockResponseWriter) WriteHeader(code int) {
	m.code = code
}

func benchRequest(b *testing.B, router http.Handler, r *http.Request) {
	w := new(mockResponseWriter)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		router.ServeHTTP(w, r)
	}
}

func serve(router *TreeMux, w http.ResponseWriter, r *http.Request, useLookup bool) bool {
	if useLookup {
		result, found := router.Lookup(w, r)
		router.ServeLookupResult(w, r, result)
		return found
	} else {
		router.ServeHTTP(w, r)
		return true
	}
}

func TestMethods(t *testing.T) {
	for _, scenario := range scenarios {
		t.Log(scenario.description)
		testMethods(t, scenario.RequestCreator, true, scenario.ServeStyle)
		testMethods(t, scenario.RequestCreator, false, scenario.ServeStyle)
	}
}

func testMethods(t *testing.T, newRequest RequestCreator, headCanUseGet bool, useSeparateLookup bool) {
	var result string

	makeHandler := func(method string) HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request, params map[string]string) {
			result = method
		}
	}

	router := New()
	router.HeadCanUseGet = headCanUseGet
	router.GET("/user/:param", makeHandler("GET"))
	router.POST("/user/:param", makeHandler("POST"))
	router.PATCH("/user/:param", makeHandler("PATCH"))
	router.PUT("/user/:param", makeHandler("PUT"))
	router.DELETE("/user/:param", makeHandler("DELETE"))

	testMethod := func(method, expect string) {
		result = ""
		w := httptest.NewRecorder()
		r, _ := newRequest(method, "/user/"+method, nil)
		found := serve(router, w, r, useSeparateLookup)

		if useSeparateLookup && expect == "" && found {
			t.Errorf("Lookup unexpectedly succeeded for method %s", method)
		}

		if expect == "" && w.Code != http.StatusMethodNotAllowed {
			t.Errorf("Method %s not expected to match but saw code %d", method, w.Code)
		}

		if result != expect {
			t.Errorf("Method %s got result %s", method, result)
		}
	}

	testMethod("GET", "GET")
	testMethod("POST", "POST")
	testMethod("PATCH", "PATCH")
	testMethod("PUT", "PUT")
	testMethod("DELETE", "DELETE")
	if headCanUseGet {
		t.Log("Test implicit HEAD with HeadCanUseGet = true")
		testMethod("HEAD", "GET")
	} else {
		t.Log("Test implicit HEAD with HeadCanUseGet = false")
		testMethod("HEAD", "")
	}

	router.HEAD("/user/:param", makeHandler("HEAD"))
	testMethod("HEAD", "HEAD")
}

func TestNotFound(t *testing.T) {
	calledNotFound := false

	notFoundHandler := func(w http.ResponseWriter, r *http.Request) {
		calledNotFound = true
	}

	router := New()
	router.GET("/user/abc", simpleHandler)

	w := httptest.NewRecorder()
	r, _ := newRequest("GET", "/abc/", nil)
	router.ServeHTTP(w, r)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected error 404 from built-in not found handler but saw %d", w.Code)
	}

	// Now try with a custome handler.
	router.NotFoundHandler = notFoundHandler

	router.ServeHTTP(w, r)
	if !calledNotFound {
		t.Error("Custom not found handler was not called")
	}
}

func TestMethodNotAllowedHandler(t *testing.T) {
	calledNotAllowed := false

	notAllowedHandler := func(w http.ResponseWriter, r *http.Request,
		methods map[string]HandlerFunc) {

		calledNotAllowed = true

		expected := []string{"GET", "PUT", "DELETE", "HEAD"}
		allowed := make([]string, 0)
		for m := range methods {
			allowed = append(allowed, m)
		}

		sort.Strings(expected)
		sort.Strings(allowed)

		if !reflect.DeepEqual(expected, allowed) {
			t.Errorf("Custom handler expected map %v, saw %v",
				expected, allowed)
		}
	}

	router := New()
	router.GET("/user/abc", simpleHandler)
	router.PUT("/user/abc", simpleHandler)
	router.DELETE("/user/abc", simpleHandler)

	w := httptest.NewRecorder()
	r, _ := newRequest("POST", "/user/abc", nil)
	router.ServeHTTP(w, r)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected error %d from built-in not found handler but saw %d",
			http.StatusMethodNotAllowed, w.Code)
	}

	allowed := w.Header()["Allow"]
	sort.Strings(allowed)
	expected := []string{"DELETE", "GET", "PUT", "HEAD"}
	sort.Strings(expected)

	if !reflect.DeepEqual(allowed, expected) {
		t.Errorf("Expected Allow header %v, saw %v",
			expected, allowed)
	}

	// Now try with a custom handler.
	router.MethodNotAllowedHandler = notAllowedHandler

	router.ServeHTTP(w, r)
	if !calledNotAllowed {
		t.Error("Custom not allowed handler was not called")
	}
}

func TestOptionsHandler(t *testing.T) {
	optionsHandler := func(w http.ResponseWriter, r *http.Request, pathParams map[string]string) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.WriteHeader(http.StatusNoContent)
	}

	customOptionsHandler := func(w http.ResponseWriter, r *http.Request, pathParams map[string]string) {
		w.Header().Set("Access-Control-Allow-Origin", "httptreemux.com")
		w.WriteHeader(http.StatusUnauthorized)
	}

	router := New()
	router.GET("/user/abc", simpleHandler)
	router.PUT("/user/abc", simpleHandler)
	router.DELETE("/user/abc", simpleHandler)
	router.OPTIONS("/user/abc/options", customOptionsHandler)

	// test without an OPTIONS handler
	w := httptest.NewRecorder()
	r, _ := newRequest("OPTIONS", "/user/abc", nil)
	router.ServeHTTP(w, r)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected error %d from built-in not found handler but saw %d",
			http.StatusMethodNotAllowed, w.Code)
	}

	// Now try with a global options handler.
	router.OptionsHandler = optionsHandler

	w = httptest.NewRecorder()
	router.ServeHTTP(w, r)
	if !(w.Code == http.StatusNoContent && w.Header()["Access-Control-Allow-Origin"][0] == "*") {
		t.Error("global options handler was not called")
	}

	// Now see if a custom handler overwrites the global options handler.
	w = httptest.NewRecorder()
	r, _ = newRequest("OPTIONS", "/user/abc/options", nil)
	router.ServeHTTP(w, r)
	if !(w.Code == http.StatusUnauthorized && w.Header()["Access-Control-Allow-Origin"][0] == "httptreemux.com") {
		t.Error("custom options handler did not overwrite global handler")
	}

	// Now see if a custom handler works with the global options handler set to nil.
	router.OptionsHandler = nil
	w = httptest.NewRecorder()
	r, _ = newRequest("OPTIONS", "/user/abc/options", nil)
	router.ServeHTTP(w, r)
	if !(w.Code == http.StatusUnauthorized && w.Header()["Access-Control-Allow-Origin"][0] == "httptreemux.com") {
		t.Error("custom options handler did not overwrite global handler")
	}

	// Make sure that the MethodNotAllowedHandler works when OptionsHandler is set
	router.OptionsHandler = optionsHandler
	w = httptest.NewRecorder()
	r, _ = newRequest("POST", "/user/abc", nil)
	router.ServeHTTP(w, r)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected error %d from built-in not found handler but saw %d",
			http.StatusMethodNotAllowed, w.Code)
	}

	allowed := w.Header()["Allow"]
	sort.Strings(allowed)
	expected := []string{"DELETE", "GET", "PUT", "HEAD"}
	sort.Strings(expected)

	if !reflect.DeepEqual(allowed, expected) {
		t.Errorf("Expected Allow header %v, saw %v",
			expected, allowed)
	}
}

func TestPanic(t *testing.T) {

	router := New()
	router.PanicHandler = SimplePanicHandler
	router.GET("/abc", panicHandler)
	r, _ := newRequest("GET", "/abc", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, r)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("Expected code %d from default panic handler, saw %d",
			http.StatusInternalServerError, w.Code)
	}

	sawPanic := false
	router.PanicHandler = func(w http.ResponseWriter, r *http.Request, err interface{}) {
		sawPanic = true
	}

	router.ServeHTTP(w, r)
	if !sawPanic {
		t.Errorf("Custom panic handler was not called")
	}

	// Assume this does the right thing. Just a sanity test.
	router.PanicHandler = ShowErrorsPanicHandler
	w = httptest.NewRecorder()
	router.ServeHTTP(w, r)
	if w.Code != http.StatusInternalServerError {
		t.Errorf("Expected code %d from ShowErrorsPanicHandler, saw %d",
			http.StatusInternalServerError, w.Code)
	}
}

func TestRedirect(t *testing.T) {
	for _, scenario := range scenarios {
		t.Log(scenario.description)
		t.Log("Testing with all 301")
		testRedirect(t, Redirect301, Redirect301, Redirect301, false, scenario.RequestCreator, scenario.ServeStyle)
		t.Log("Testing with all UseHandler")
		testRedirect(t, UseHandler, UseHandler, UseHandler, false, scenario.RequestCreator, scenario.ServeStyle)
		t.Log("Testing with default 301, GET 307, POST UseHandler")
		testRedirect(t, Redirect301, Redirect307, UseHandler, true, scenario.RequestCreator, scenario.ServeStyle)
		t.Log("Testing with default UseHandler, GET 301, POST 308")
		testRedirect(t, UseHandler, Redirect301, Redirect308, true, scenario.RequestCreator, scenario.ServeStyle)
	}
}

func behaviorToCode(b RedirectBehavior) int {
	switch b {
	case Redirect301:
		return http.StatusMovedPermanently
	case Redirect307:
		return http.StatusTemporaryRedirect
	case Redirect308:
		return 308
	case UseHandler:
		// Not normally, but the handler in the below test returns this.
		return http.StatusNoContent
	}

	panic("Unhandled behavior!")
}

func testRedirect(t *testing.T, defaultBehavior, getBehavior, postBehavior RedirectBehavior, customMethods bool,
	newRequest RequestCreator, serveStyle bool) {

	var redirHandler = func(w http.ResponseWriter, r *http.Request, params map[string]string) {
		// Returning this instead of 200 makes it easy to verify that the handler is actually getting called.
		w.WriteHeader(http.StatusNoContent)
	}

	router := New()
	router.RedirectBehavior = defaultBehavior

	var expectedCodeMap = map[string]int{"PUT": behaviorToCode(defaultBehavior)}

	if customMethods {
		router.RedirectMethodBehavior["GET"] = getBehavior
		router.RedirectMethodBehavior["POST"] = postBehavior
		expectedCodeMap["GET"] = behaviorToCode(getBehavior)
		expectedCodeMap["POST"] = behaviorToCode(postBehavior)
	} else {
		expectedCodeMap["GET"] = expectedCodeMap["PUT"]
		expectedCodeMap["POST"] = expectedCodeMap["PUT"]
	}

	router.GET("/slash/", redirHandler)
	router.GET("/noslash", redirHandler)
	router.POST("/slash/", redirHandler)
	router.POST("/noslash", redirHandler)
	router.PUT("/slash/", redirHandler)
	router.PUT("/noslash", redirHandler)

	for method, expectedCode := range expectedCodeMap {
		t.Logf("Testing method %s, expecting code %d", method, expectedCode)

		w := httptest.NewRecorder()
		r, _ := newRequest(method, "/slash", nil)
		found := serve(router, w, r, serveStyle)
		if found == false {
			t.Errorf("/slash: found returned false")
		}
		if w.Code != expectedCode {
			t.Errorf("/slash expected code %d, saw %d", expectedCode, w.Code)
		}
		if expectedCode != http.StatusNoContent && w.Header().Get("Location") != "/slash/" {
			t.Errorf("/slash was not redirected to /slash/")
		}

		r, _ = newRequest(method, "/noslash/", nil)
		w = httptest.NewRecorder()
		found = serve(router, w, r, serveStyle)
		if found == false {
			t.Errorf("/noslash: found returned false")
		}
		if w.Code != expectedCode {
			t.Errorf("/noslash/ expected code %d, saw %d", expectedCode, w.Code)
		}
		if expectedCode != http.StatusNoContent && w.Header().Get("Location") != "/noslash" {
			t.Errorf("/noslash/ was redirected to `%s` instead of /noslash", w.Header().Get("Location"))
		}

		r, _ = newRequest(method, "//noslash/", nil)
		if r.RequestURI == "//noslash/" { // http.NewRequest parses this out differently
			w = httptest.NewRecorder()
			found = serve(router, w, r, serveStyle)
			if found == false {
				t.Errorf("//noslash/: found returned false")
			}
			if w.Code != expectedCode {
				t.Errorf("//noslash/ expected code %d, saw %d", expectedCode, w.Code)
			}
			if expectedCode != http.StatusNoContent && w.Header().Get("Location") != "/noslash" {
				t.Errorf("//noslash/ was redirected to %s, expected /noslash", w.Header().Get("Location"))
			}
		}

		// Test nonredirect cases
		r, _ = newRequest(method, "/noslash", nil)
		w = httptest.NewRecorder()
		found = serve(router, w, r, serveStyle)
		if found == false {
			t.Errorf("/noslash (non-redirect): found returned false")
		}
		if w.Code != http.StatusNoContent {
			t.Errorf("/noslash (non-redirect) expected code %d, saw %d", http.StatusNoContent, w.Code)
		}

		r, _ = newRequest(method, "/noslash?a=1&b=2", nil)
		w = httptest.NewRecorder()
		found = serve(router, w, r, serveStyle)
		if found == false {
			t.Errorf("/noslash (non-redirect): found returned false")
		}
		if w.Code != http.StatusNoContent {
			t.Errorf("/noslash (non-redirect) expected code %d, saw %d", http.StatusNoContent, w.Code)
		}

		r, _ = newRequest(method, "/slash/", nil)
		w = httptest.NewRecorder()
		found = serve(router, w, r, serveStyle)
		if found == false {
			t.Errorf("/slash/ (non-redirect): found returned false")
		}
		if w.Code != http.StatusNoContent {
			t.Errorf("/slash/ (non-redirect) expected code %d, saw %d", http.StatusNoContent, w.Code)
		}

		r, _ = newRequest(method, "/slash/?a=1&b=2", nil)
		w = httptest.NewRecorder()
		found = serve(router, w, r, serveStyle)
		if found == false {
			t.Errorf("/slash/?a=1&b=2: found returned false")
		}
		if w.Code != http.StatusNoContent {
			t.Errorf("/slash/?a=1&b=2 expected code %d, saw %d", http.StatusNoContent, w.Code)
		}

		// Test querystring and fragment cases
		r, _ = newRequest(method, "/slash?a=1&b=2", nil)
		w = httptest.NewRecorder()
		found = serve(router, w, r, serveStyle)
		if found == false {
			t.Errorf("/slash?a=1&b=2 : found returned false")
		}
		if w.Code != expectedCode {
			t.Errorf("/slash?a=1&b=2 expected code %d, saw %d", expectedCode, w.Code)
		}
		if expectedCode != http.StatusNoContent && w.Header().Get("Location") != "/slash/?a=1&b=2" {
			t.Errorf("/slash?a=1&b=2 was redirected to %s", w.Header().Get("Location"))
		}

		r, _ = newRequest(method, "/noslash/?a=1&b=2", nil)
		w = httptest.NewRecorder()
		found = serve(router, w, r, serveStyle)
		if found == false {
			t.Errorf("/noslash/?a=1&b=2: found returned false")
		}
		if w.Code != expectedCode {
			t.Errorf("/noslash/?a=1&b=2 expected code %d, saw %d", expectedCode, w.Code)
		}
		if expectedCode != http.StatusNoContent && w.Header().Get("Location") != "/noslash?a=1&b=2" {
			t.Errorf("/noslash/?a=1&b=2 was redirected to %s", w.Header().Get("Location"))
		}
	}
}

func TestSkipRedirect(t *testing.T) {
	router := New()
	router.RedirectTrailingSlash = false
	router.RedirectCleanPath = false
	router.GET("/slash/", simpleHandler)
	router.GET("/noslash", simpleHandler)

	w := httptest.NewRecorder()
	r, _ := newRequest("GET", "/slash", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("/slash expected code 404, saw %d", w.Code)
	}

	r, _ = newRequest("GET", "/noslash/", nil)
	w = httptest.NewRecorder()
	router.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("/noslash/ expected code 404, saw %d", w.Code)
	}

	r, _ = newRequest("GET", "//noslash", nil)
	w = httptest.NewRecorder()
	router.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("//noslash expected code 404, saw %d", w.Code)
	}
}

func TestCatchAllTrailingSlashRedirect(t *testing.T) {
	router := New()
	redirectSettings := []bool{false, true}

	router.GET("/abc/*path", simpleHandler)

	testPath := func(path string) {
		r, _ := newRequest("GET", "/abc/"+path, nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, r)

		endingSlash := strings.HasSuffix(path, "/")

		var expectedCode int
		if endingSlash && router.RedirectTrailingSlash && router.RemoveCatchAllTrailingSlash {
			expectedCode = http.StatusMovedPermanently
		} else {
			expectedCode = http.StatusOK
		}

		if w.Code != expectedCode {
			t.Errorf("Path %s with RedirectTrailingSlash %v, RemoveCatchAllTrailingSlash %v "+
				" expected code %d but saw %d", path,
				router.RedirectTrailingSlash, router.RemoveCatchAllTrailingSlash,
				expectedCode, w.Code)
		}
	}

	for _, redirectSetting := range redirectSettings {
		for _, removeCatchAllSlash := range redirectSettings {
			router.RemoveCatchAllTrailingSlash = removeCatchAllSlash
			router.RedirectTrailingSlash = redirectSetting

			testPath("apples")
			testPath("apples/")
			testPath("apples/bananas")
			testPath("apples/bananas/")
		}
	}

}

func TestRoot(t *testing.T) {
	for _, scenario := range scenarios {
		t.Log(scenario.description)
		handlerCalled := false
		handler := func(w http.ResponseWriter, r *http.Request, params map[string]string) {
			handlerCalled = true
		}
		router := New()
		router.GET("/", handler)

		r, _ := scenario.RequestCreator("GET", "/", nil)
		w := new(mockResponseWriter)
		serve(router, w, r, scenario.ServeStyle)

		if !handlerCalled {
			t.Error("Handler not called for root path")
		}
	}
}

func TestWildcardAtSplitNode(t *testing.T) {
	var suppliedParam string
	simpleHandler := func(w http.ResponseWriter, r *http.Request, params map[string]string) {
		t.Log(params)
		suppliedParam, _ = params["slug"]
	}

	router := New()
	router.GET("/pumpkin", simpleHandler)
	router.GET("/passing", simpleHandler)
	router.GET("/:slug", simpleHandler)
	router.GET("/:slug/abc", simpleHandler)

	t.Log(router.root.dumpTree("", " "))

	r, _ := newRequest("GET", "/patch", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, r)

	if suppliedParam != "patch" {
		t.Errorf("Expected param patch, saw %s", suppliedParam)
	}

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200 for path /patch, saw %d", w.Code)
	}

	suppliedParam = ""
	r, _ = newRequest("GET", "/patch/abc", nil)
	w = httptest.NewRecorder()
	router.ServeHTTP(w, r)

	if suppliedParam != "patch" {
		t.Errorf("Expected param patch, saw %s", suppliedParam)
	}

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200 for path /patch/abc, saw %d", w.Code)
	}

	r, _ = newRequest("GET", "/patch/def", nil)
	w = httptest.NewRecorder()
	router.ServeHTTP(w, r)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected status 404 for path /patch/def, saw %d", w.Code)
	}
}

func TestSlash(t *testing.T) {
	param := ""
	handler := func(w http.ResponseWriter, r *http.Request, params map[string]string) {
		param = params["param"]
	}
	ymHandler := func(w http.ResponseWriter, r *http.Request, params map[string]string) {
		param = params["year"] + " " + params["month"]
	}
	router := New()
	router.GET("/abc/:param", handler)
	router.GET("/year/:year/month/:month", ymHandler)

	r, _ := newRequest("GET", "/abc/de%2ff", nil)
	w := new(mockResponseWriter)
	router.ServeHTTP(w, r)

	if param != "de/f" {
		t.Errorf("Expected param de/f, saw %s", param)
	}

	r, _ = newRequest("GET", "/year/de%2f/month/fg%2f", nil)
	router.ServeHTTP(w, r)

	if param != "de/ fg/" {
		t.Errorf("Expected param de/ fg/, saw %s", param)
	}
}

func TestQueryString(t *testing.T) {
	for _, scenario := range scenarios {
		t.Log(scenario.description)
		param := ""
		handler := func(w http.ResponseWriter, r *http.Request, params map[string]string) {
			param = params["param"]
		}
		router := New()
		router.GET("/static", handler)
		router.GET("/wildcard/:param", handler)
		router.GET("/catchall/*param", handler)

		r, _ := scenario.RequestCreator("GET", "/static?abc=def&ghi=jkl", nil)
		w := new(mockResponseWriter)

		param = "nomatch"
		serve(router, w, r, scenario.ServeStyle)
		if param != "" {
			t.Error("No match on", r.RequestURI)
		}

		r, _ = scenario.RequestCreator("GET", "/wildcard/aaa?abc=def", nil)
		serve(router, w, r, scenario.ServeStyle)
		if param != "aaa" {
			t.Error("Expected wildcard to match aaa, saw", param)
		}

		r, _ = scenario.RequestCreator("GET", "/catchall/bbb?abc=def", nil)
		serve(router, w, r, scenario.ServeStyle)
		if param != "bbb" {
			t.Error("Expected wildcard to match bbb, saw", param)
		}
	}
}

func TestPathSource(t *testing.T) {
	var called string

	appleHandler := func(w http.ResponseWriter, r *http.Request, params map[string]string) {
		called = "apples"
	}

	bananaHandler := func(w http.ResponseWriter, r *http.Request, params map[string]string) {
		called = "bananas"
	}
	router := New()
	router.GET("/apples", appleHandler)
	router.GET("/bananas", bananaHandler)

	// Set up a request with different values in URL and RequestURI.
	r, _ := newRequest("GET", "/apples", nil)
	r.RequestURI = "/bananas"
	w := new(mockResponseWriter)

	// Default setting should be RequestURI
	router.ServeHTTP(w, r)
	if called != "bananas" {
		t.Error("Using default, expected bananas but saw", called)
	}

	router.PathSource = URLPath
	router.ServeHTTP(w, r)
	if called != "apples" {
		t.Error("Using URLPath, expected apples but saw", called)
	}

	router.PathSource = RequestURI
	router.ServeHTTP(w, r)
	if called != "bananas" {
		t.Error("Using RequestURI, expected bananas but saw", called)
	}
}

func TestEscapedRoutes(t *testing.T) {
	type testcase struct {
		Route      string
		Path       string
		Param      string
		ParamValue string
	}

	testcases := []*testcase{
		{"/abc/def", "/abc/def", "", ""},
		{"/abc/*star", "/abc/defg", "star", "defg"},
		{"/abc/extrapath/*star", "/abc/extrapath/*lll", "star", "*lll"},
		{"/abc/\\*def", "/abc/*def", "", ""},
		{"/abc/\\\\*def", "/abc/\\*def", "", ""},
		{"/:wild/def", "/*abcd/def", "wild", "*abcd"},
		{"/\\:wild/def", "/:wild/def", "", ""},
		{"/\\\\:wild/def", "/\\:wild/def", "", ""},
		{"/\\*abc/def", "/*abc/def", "", ""},
	}

	escapeCases := []bool{false, true}

	for _, escape := range escapeCases {
		var foundTestCase *testcase
		var foundParamKey string
		var foundParamValue string

		handleTestResponse := func(c *testcase, w http.ResponseWriter, r *http.Request, params map[string]string) {
			foundTestCase = c
			foundParamKey = ""
			foundParamValue = ""
			for key, val := range params {
				foundParamKey = key
				foundParamValue = val
			}
			t.Logf("RequestURI %s found test case %+v", r.RequestURI, c)
		}

		verify := func(c *testcase) {
			t.Logf("Expecting test case %+v", c)
			if c != foundTestCase {
				t.Errorf("Incorrectly matched test case %+v", foundTestCase)
			}

			if c.Param != foundParamKey {
				t.Errorf("Expected param key %s but saw %s", c.Param, foundParamKey)
			}

			if c.ParamValue != foundParamValue {
				t.Errorf("Expected param key %s but saw %s", c.Param, foundParamKey)
			}
		}

		t.Log("Recreating router")
		router := New()
		router.EscapeAddedRoutes = escape

		for _, c := range testcases {
			t.Logf("Adding route %s", c.Route)
			theCase := c
			router.GET(c.Route, func(w http.ResponseWriter, r *http.Request, params map[string]string) {
				handleTestResponse(theCase, w, r, params)
			})
		}

		for _, c := range testcases {
			escapedPath := (&url.URL{Path: c.Path}).EscapedPath()
			escapedIsSame := escapedPath == c.Path

			r, _ := newRequest("GET", c.Path, nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, r)
			if w.Code != 200 {
				t.Errorf("Escape %v test case %v saw code %d", escape, c, w.Code)
			}
			verify(c)

			if !escapedIsSame {
				r, _ := newRequest("GET", escapedPath, nil)
				w := httptest.NewRecorder()
				router.ServeHTTP(w, r)
				if router.EscapeAddedRoutes {
					// Expect a match
					if w.Code != 200 {
						t.Errorf("Escape %v test case %v saw code %d", escape, c, w.Code)
					}
					verify(c)
				} else {
					// Expect a non-match if the parameter isn't a wildcard.
					if foundParamKey == "" && w.Code != 404 {
						t.Errorf("Escape %v test case %v expected 404 saw %d", escape, c, w.Code)
					}
				}
			}
		}
	}
}

// Create a bunch of paths for testing.
func createRoutes(numRoutes int) []string {
	letters := "abcdefghijhklmnopqrstuvwxyz"
	wordMap := map[string]bool{}
	for i := 0; i < numRoutes/2; i += 1 {
		length := (i % 4) + 4

		wordBytes := make([]byte, length)
		for charIndex := 0; charIndex < length; charIndex += 1 {
			wordBytes[charIndex] = letters[(i*3+charIndex*4)%len(letters)]
		}
		wordMap[string(wordBytes)] = true
	}

	words := make([]string, 0, len(wordMap))
	for word := range wordMap {
		words = append(words, word)
	}

	routes := make([]string, 0, numRoutes)
	createdRoutes := map[string]bool{}
	rand.Seed(0)
	for len(routes) < numRoutes {
		first := words[rand.Int()%len(words)]
		second := words[rand.Int()%len(words)]
		third := words[rand.Int()%len(words)]
		route := fmt.Sprintf("/%s/%s/%s", first, second, third)

		if createdRoutes[route] {
			continue
		}
		createdRoutes[route] = true
		routes = append(routes, route)
	}

	return routes
}

// TestWriteConcurrency ensures that the router works with multiple goroutines adding
// routes concurrently.
func TestWriteConcurrency(t *testing.T) {
	router := New()

	// First create a bunch of routes
	numRoutes := 10000
	routes := createRoutes(numRoutes)

	wg := sync.WaitGroup{}
	addRoutes := func(base int, method string) {
		for i := 0; i < len(routes); i += 1 {
			route := routes[(i+base)%len(routes)]
			// t.Logf("Adding %s %s", method, route)
			router.Handle(method, route, simpleHandler)
		}
		wg.Done()
	}

	wg.Add(5)
	go addRoutes(100, "GET")
	go addRoutes(200, "POST")
	go addRoutes(300, "PATCH")
	go addRoutes(400, "PUT")
	go addRoutes(500, "DELETE")
	wg.Wait()

	handleRequests := func(method string) {
		for _, route := range routes {
			// t.Logf("Serving %s %s", method, route)
			r, _ := newRequest(method, route, nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, r)

			if w.Code != 200 {
				t.Errorf("%s %s request failed", method, route)
			}
		}
	}

	handleRequests("GET")
	handleRequests("POST")
	handleRequests("PATCH")
	handleRequests("PUT")
	handleRequests("DELETE")
}

// TestReadWriteConcurrency ensures that when SafeAddRoutesWhileRunning is enabled,
// the router is able to add routes while serving traffic.
func TestReadWriteConcurrency(t *testing.T) {
	router := New()
	router.SafeAddRoutesWhileRunning = true

	// First create a bunch of routes
	numRoutes := 10000
	routes := createRoutes(numRoutes)

	wg := sync.WaitGroup{}
	addRoutes := func(base int, method string, routes []string) {
		for i := 0; i < len(routes); i += 1 {
			route := routes[(i+base)%len(routes)]
			// t.Logf("Adding %s %s", method, route)
			router.Handle(method, route, simpleHandler)
		}
		wg.Done()
	}

	handleRequests := func(base int, method string, routes []string, requireFound bool) {
		for i := 0; i < len(routes); i += 1 {
			route := routes[(i+base)%len(routes)]
			// t.Logf("Serving %s %s", method, route)
			r, _ := newRequest(method, route, nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, r)

			if requireFound && w.Code != 200 {
				t.Errorf("%s %s request failed", method, route)
			}
		}
		wg.Done()
	}

	wg.Add(12)
	initialRoutes := routes[0 : numRoutes/10]
	addRoutes(0, "GET", initialRoutes)
	handleRequests(0, "GET", initialRoutes, true)

	concurrentRoutes := routes[numRoutes/10:]
	go addRoutes(100, "GET", concurrentRoutes)
	go addRoutes(200, "POST", concurrentRoutes)
	go addRoutes(300, "PATCH", concurrentRoutes)
	go addRoutes(400, "PUT", concurrentRoutes)
	go addRoutes(500, "DELETE", concurrentRoutes)
	go handleRequests(50, "GET", routes, false)
	go handleRequests(150, "POST", routes, false)
	go handleRequests(250, "PATCH", routes, false)
	go handleRequests(350, "PUT", routes, false)
	go handleRequests(450, "DELETE", routes, false)

	wg.Wait()

	// Finally check all the routes and make sure they exist.
	wg.Add(5)
	handleRequests(0, "GET", routes, true)
	handleRequests(0, "POST", concurrentRoutes, true)
	handleRequests(0, "PATCH", concurrentRoutes, true)
	handleRequests(0, "PUT", concurrentRoutes, true)
	handleRequests(0, "DELETE", concurrentRoutes, true)
}

func TestLookup(t *testing.T) {
	router := New()
	router.GET("/", simpleHandler)
	router.GET("/user/dimfeld", simpleHandler)
	router.POST("/user/dimfeld", simpleHandler)
	router.GET("/abc/*", simpleHandler)
	router.POST("/abc/*", simpleHandler)

	var tryLookup = func(method, path string, expectFound bool, expectCode int) {
		r, _ := newRequest(method, path, nil)
		w := &mockResponseWriter{}
		lr, found := router.Lookup(w, r)
		if found != expectFound {
			t.Errorf("%s %s expected found %v, saw %v", method, path, expectFound, found)
		}

		if lr.StatusCode != expectCode {
			t.Errorf("%s %s expected status code %d, saw %d", method, path, expectCode, lr.StatusCode)
		}

		if w.code != 0 {
			t.Errorf("%s %s unexpectedly wrote status %d", method, path, w.code)
		}

		if w.calledWrite {
			t.Errorf("%s %s unexpectedly wrote data", method, path)
		}
	}

	tryLookup("GET", "/", true, http.StatusOK)
	tryLookup("GET", "/", true, http.StatusOK)
	tryLookup("POST", "/user/dimfeld", true, http.StatusOK)
	tryLookup("POST", "/user/dimfeld/", true, http.StatusMovedPermanently)
	tryLookup("PATCH", "/user/dimfeld", false, http.StatusMethodNotAllowed)
	tryLookup("GET", "/abc/def/ghi", true, http.StatusOK)

	router.RedirectBehavior = Redirect307
	tryLookup("POST", "/user/dimfeld/", true, http.StatusTemporaryRedirect)
}

func TestRedirectEscapedPath(t *testing.T) {
	router := New()

	testHandler := func(w http.ResponseWriter, r *http.Request, params map[string]string) {}

	router.GET("/:escaped/", testHandler)

	w := httptest.NewRecorder()
	u, err := url.Parse("/Test P@th")
	if err != nil {
		t.Error(err)
		return
	}

	r, _ := newRequest("GET", u.String(), nil)

	router.ServeHTTP(w, r)

	if w.Code != http.StatusMovedPermanently {
		t.Errorf("Expected status 301 but saw %d", w.Code)
	}

	path := w.Header().Get("Location")
	expected := "/Test%20P@th/"
	if path != expected {
		t.Errorf("Given path wasn't escaped correctly.\n"+
			"Expected: %q\nBut got: %q", expected, path)
	}
}

func BenchmarkRouterSimple(b *testing.B) {
	router := New()

	router.GET("/", simpleHandler)
	router.GET("/user/dimfeld", simpleHandler)

	r, _ := newRequest("GET", "/user/dimfeld", nil)

	benchRequest(b, router, r)
}

func BenchmarkRouterRootWithPanicHandler(b *testing.B) {
	router := New()
	router.PanicHandler = SimplePanicHandler

	router.GET("/", simpleHandler)
	router.GET("/user/dimfeld", simpleHandler)

	r, _ := newRequest("GET", "/", nil)

	benchRequest(b, router, r)
}

func BenchmarkRouterRootWithoutPanicHandler(b *testing.B) {
	router := New()
	router.PanicHandler = nil

	router.GET("/", simpleHandler)
	router.GET("/user/dimfeld", simpleHandler)

	r, _ := newRequest("GET", "/", nil)

	benchRequest(b, router, r)
}

func BenchmarkRouterParam(b *testing.B) {
	router := New()

	router.GET("/", simpleHandler)
	router.GET("/user/:name", simpleHandler)

	r, _ := newRequest("GET", "/user/dimfeld", nil)

	benchRequest(b, router, r)
}

func BenchmarkRouterLongParams(b *testing.B) {
	router := New()

	router.GET("/", simpleHandler)
	router.GET("/user/:name/:resource", simpleHandler)

	r, _ := newRequest("GET", "/user/aaaabbbbccccddddeeeeffff/asdfghjkl", nil)

	benchRequest(b, router, r)
}
