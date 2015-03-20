package restful

import (
	"io"
	"sort"
	"testing"
)

//
// Step 1 tests
//
var paths = []struct {
	// url with path (1) is handled by service with root (2) and last capturing group has value final (3)
	path, root, final string
}{
	{"/", "/", "/"},
	{"/p", "/p", ""},
	{"/p/x", "/p/{q}", ""},
	{"/q/x", "/q", "/x"},
	{"/p/x/", "/p/{q}", "/"},
	{"/p/x/y", "/p/{q}", "/y"},
	{"/q/x/y", "/q", "/x/y"},
	{"/z/q", "/{p}/q", ""},
	{"/a/b/c/q", "/", "/a/b/c/q"},
}

func TestDetectDispatcher(t *testing.T) {
	ws1 := new(WebService).Path("/")
	ws2 := new(WebService).Path("/p")
	ws3 := new(WebService).Path("/q")
	ws4 := new(WebService).Path("/p/q")
	ws5 := new(WebService).Path("/p/{q}")
	ws6 := new(WebService).Path("/p/{q}/")
	ws7 := new(WebService).Path("/{p}/q")
	var dispatchers = []*WebService{ws1, ws2, ws3, ws4, ws5, ws6, ws7}

	wc := NewContainer()
	for _, each := range dispatchers {
		wc.Add(each)
	}

	router := RouterJSR311{}

	ok := true
	for i, fixture := range paths {
		who, final, err := router.detectDispatcher(fixture.path, dispatchers)
		if err != nil {
			t.Logf("error in detection:%v", err)
			ok = false
		}
		if who.RootPath() != fixture.root {
			t.Logf("[line:%v] Unexpected dispatcher, expected:%v, actual:%v", i, fixture.root, who.RootPath())
			ok = false
		}
		if final != fixture.final {
			t.Logf("[line:%v] Unexpected final, expected:%v, actual:%v", i, fixture.final, final)
			ok = false
		}
	}
	if !ok {
		t.Fail()
	}
}

//
// Step 2 tests
//

// go test -v -test.run TestISSUE_179 ...restful
func TestISSUE_179(t *testing.T) {
	ws1 := new(WebService)
	ws1.Route(ws1.GET("/v1/category/{param:*}").To(dummy))
	routes := RouterJSR311{}.selectRoutes(ws1, "/v1/category/sub/sub")
	t.Logf("%v", routes)
}

// go test -v -test.run TestISSUE_30 ...restful
func TestISSUE_30(t *testing.T) {
	ws1 := new(WebService).Path("/users")
	ws1.Route(ws1.GET("/{id}").To(dummy))
	ws1.Route(ws1.POST("/login").To(dummy))
	routes := RouterJSR311{}.selectRoutes(ws1, "/login")
	if len(routes) != 2 {
		t.Fatal("expected 2 routes")
	}
	if routes[0].Path != "/users/login" {
		t.Error("first is", routes[0].Path)
		t.Logf("routes:%v", routes)
	}
}

// go test -v -test.run TestISSUE_34 ...restful
func TestISSUE_34(t *testing.T) {
	ws1 := new(WebService).Path("/")
	ws1.Route(ws1.GET("/{type}/{id}").To(dummy))
	ws1.Route(ws1.GET("/network/{id}").To(dummy))
	routes := RouterJSR311{}.selectRoutes(ws1, "/network/12")
	if len(routes) != 2 {
		t.Fatal("expected 2 routes")
	}
	if routes[0].Path != "/network/{id}" {
		t.Error("first is", routes[0].Path)
		t.Logf("routes:%v", routes)
	}
}

// go test -v -test.run TestISSUE_34_2 ...restful
func TestISSUE_34_2(t *testing.T) {
	ws1 := new(WebService).Path("/")
	// change the registration order
	ws1.Route(ws1.GET("/network/{id}").To(dummy))
	ws1.Route(ws1.GET("/{type}/{id}").To(dummy))
	routes := RouterJSR311{}.selectRoutes(ws1, "/network/12")
	if len(routes) != 2 {
		t.Fatal("expected 2 routes")
	}
	if routes[0].Path != "/network/{id}" {
		t.Error("first is", routes[0].Path)
	}
}

// go test -v -test.run TestISSUE_137 ...restful
func TestISSUE_137(t *testing.T) {
	ws1 := new(WebService)
	ws1.Route(ws1.GET("/hello").To(dummy))
	routes := RouterJSR311{}.selectRoutes(ws1, "/")
	t.Log(routes)
	if len(routes) > 0 {
		t.Error("no route expected")
	}
}

func TestSelectRoutesSlash(t *testing.T) {
	ws1 := new(WebService).Path("/")
	ws1.Route(ws1.GET("").To(dummy))
	ws1.Route(ws1.GET("/").To(dummy))
	ws1.Route(ws1.GET("/u").To(dummy))
	ws1.Route(ws1.POST("/u").To(dummy))
	ws1.Route(ws1.POST("/u/v").To(dummy))
	ws1.Route(ws1.POST("/u/{w}").To(dummy))
	ws1.Route(ws1.POST("/u/{w}/z").To(dummy))
	routes := RouterJSR311{}.selectRoutes(ws1, "/u")
	checkRoutesContains(routes, "/u", t)
	checkRoutesContainsNo(routes, "/u/v", t)
	checkRoutesContainsNo(routes, "/", t)
	checkRoutesContainsNo(routes, "/u/{w}/z", t)
}
func TestSelectRoutesU(t *testing.T) {
	ws1 := new(WebService).Path("/u")
	ws1.Route(ws1.GET("").To(dummy))
	ws1.Route(ws1.GET("/").To(dummy))
	ws1.Route(ws1.GET("/v").To(dummy))
	ws1.Route(ws1.POST("/{w}").To(dummy))
	ws1.Route(ws1.POST("/{w}/z").To(dummy))          // so full path = /u/{w}/z
	routes := RouterJSR311{}.selectRoutes(ws1, "/v") // test against /u/v
	checkRoutesContains(routes, "/u/{w}", t)
}

func TestSelectRoutesUsers1(t *testing.T) {
	ws1 := new(WebService).Path("/users")
	ws1.Route(ws1.POST("").To(dummy))
	ws1.Route(ws1.POST("/").To(dummy))
	ws1.Route(ws1.PUT("/{id}").To(dummy))
	routes := RouterJSR311{}.selectRoutes(ws1, "/1")
	checkRoutesContains(routes, "/users/{id}", t)
}
func checkRoutesContains(routes []Route, path string, t *testing.T) {
	if !containsRoutePath(routes, path, t) {
		for _, r := range routes {
			t.Logf("route %v %v", r.Method, r.Path)
		}
		t.Fatalf("routes should include [%v]:", path)
	}
}
func checkRoutesContainsNo(routes []Route, path string, t *testing.T) {
	if containsRoutePath(routes, path, t) {
		for _, r := range routes {
			t.Logf("route %v %v", r.Method, r.Path)
		}
		t.Fatalf("routes should not include [%v]:", path)
	}
}
func containsRoutePath(routes []Route, path string, t *testing.T) bool {
	for _, each := range routes {
		if each.Path == path {
			return true
		}
	}
	return false
}

// go test -v -test.run TestSortableRouteCandidates ...restful
func TestSortableRouteCandidates(t *testing.T) {
	fixture := &sortableRouteCandidates{}
	r1 := routeCandidate{matchesCount: 0, literalCount: 0, nonDefaultCount: 0}
	r2 := routeCandidate{matchesCount: 0, literalCount: 0, nonDefaultCount: 1}
	r3 := routeCandidate{matchesCount: 0, literalCount: 1, nonDefaultCount: 1}
	r4 := routeCandidate{matchesCount: 1, literalCount: 1, nonDefaultCount: 0}
	r5 := routeCandidate{matchesCount: 1, literalCount: 0, nonDefaultCount: 0}
	fixture.candidates = append(fixture.candidates, r5, r4, r3, r2, r1)
	sort.Sort(sort.Reverse(fixture))
	first := fixture.candidates[0]
	if first.matchesCount != 1 && first.literalCount != 1 && first.nonDefaultCount != 0 {
		t.Fatal("expected r4")
	}
	last := fixture.candidates[len(fixture.candidates)-1]
	if last.matchesCount != 0 && last.literalCount != 0 && last.nonDefaultCount != 0 {
		t.Fatal("expected r1")
	}
}

func dummy(req *Request, resp *Response) { io.WriteString(resp.ResponseWriter, "dummy") }
