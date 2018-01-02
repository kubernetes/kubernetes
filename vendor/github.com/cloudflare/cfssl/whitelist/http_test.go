package whitelist

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

type testHandler struct {
	Message string
}

func newTestHandler(m string) http.Handler {
	return &testHandler{Message: m}
}

func (h *testHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte(h.Message))
}

var testAllowHandler = newTestHandler("OK")
var testDenyHandler = newTestHandler("NO")

func testHTTPResponse(url string, t *testing.T) string {
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("%v", err)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("%v", err)
	}
	resp.Body.Close()
	return string(body)
}

func testWorker(url string, t *testing.T, wg *sync.WaitGroup) {
	for i := 0; i < 100; i++ {
		response := testHTTPResponse(url, t)
		if response != "NO" {
			t.Fatalf("Expected NO, but got %s", response)
		}
	}
	wg.Done()
}

func TestHostStubHTTP(t *testing.T) {
	wl := NewHostStub()
	h, err := NewHandler(testAllowHandler, testDenyHandler, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	response := testHTTPResponse(srv.URL, t)
	if response != "OK" {
		t.Fatalf("Expected OK, but got %s", response)
	}

	addIPString(wl, "127.0.0.1", t)
	response = testHTTPResponse(srv.URL, t)
	if response != "OK" {
		t.Fatalf("Expected OK, but got %s", response)
	}

	delIPString(wl, "127.0.0.1", t)
	response = testHTTPResponse(srv.URL, t)
	if response != "OK" {
		t.Fatalf("Expected OK, but got %s", response)
	}
}

func TestNetStubHTTP(t *testing.T) {
	wl := NewNetStub()
	h, err := NewHandler(testAllowHandler, testDenyHandler, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	response := testHTTPResponse(srv.URL, t)
	if response != "OK" {
		t.Fatalf("Expected OK, but got %s", response)
	}

	testAddNet(wl, "127.0.0.1/32", t)
	response = testHTTPResponse(srv.URL, t)
	if response != "OK" {
		t.Fatalf("Expected OK, but got %s", response)
	}

	testDelNet(wl, "127.0.0.1/32", t)
	response = testHTTPResponse(srv.URL, t)
	if response != "OK" {
		t.Fatalf("Expected OK, but got %s", response)
	}
}

func TestBasicHTTP(t *testing.T) {
	wl := NewBasic()
	h, err := NewHandler(testAllowHandler, testDenyHandler, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	response := testHTTPResponse(srv.URL, t)
	if response != "NO" {
		t.Fatalf("Expected NO, but got %s", response)
	}

	addIPString(wl, "127.0.0.1", t)
	response = testHTTPResponse(srv.URL, t)
	if response != "OK" {
		t.Fatalf("Expected OK, but got %s", response)
	}

	delIPString(wl, "127.0.0.1", t)
	response = testHTTPResponse(srv.URL, t)
	if response != "NO" {
		t.Fatalf("Expected NO, but got %s", response)
	}
}

func TestBasicHTTPDefaultDeny(t *testing.T) {
	wl := NewBasic()
	h, err := NewHandler(testAllowHandler, nil, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	expected := "Unauthorized"
	response := strings.TrimSpace(testHTTPResponse(srv.URL, t))
	if response != expected {
		t.Fatalf("Expected %s, but got %s", expected, response)
	}
}

func TestBasicHTTPWorkers(t *testing.T) {
	wl := NewBasic()
	h, err := NewHandler(testAllowHandler, testDenyHandler, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	wg := new(sync.WaitGroup)
	defer srv.Close()

	for i := 0; i < 16; i++ {
		wg.Add(1)
		go testWorker(srv.URL, t, wg)
	}

	wg.Wait()

}

func TestFailHTTP(t *testing.T) {
	wl := NewBasic()
	h, err := NewHandler(testAllowHandler, testDenyHandler, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	w := httptest.NewRecorder()
	req := new(http.Request)

	if h.ServeHTTP(w, req); w.Code != http.StatusInternalServerError {
		t.Fatalf("Expect HTTP 500, but got HTTP %d", w.Code)
	}
}

var testHandlerFunc *HandlerFunc

func newTestHandlerFunc(m string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(m))
	}
}

var testAllowHandlerFunc = newTestHandlerFunc("OK")
var testDenyHandlerFunc = newTestHandlerFunc("NO")

func TestSetupHandlerFuncFails(t *testing.T) {
	wl := NewBasic()
	_, err := NewHandlerFunc(nil, testDenyHandlerFunc, wl)
	if err == nil {
		t.Fatal("expected NewHandlerFunc to fail with nil allow handler")
	}

	_, err = NewHandlerFunc(testAllowHandlerFunc, testDenyHandlerFunc, nil)
	if err == nil {
		t.Fatal("expected NewHandlerFunc to fail with nil whitelist")
	}

	_, err = NewHandlerFunc(testAllowHandlerFunc, nil, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}
}

func TestSetupHandlerFunc(t *testing.T) {
	wl := NewBasic()
	h, err := NewHandlerFunc(testAllowHandlerFunc, testDenyHandlerFunc, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	expected := "NO"
	response := strings.TrimSpace(testHTTPResponse(srv.URL, t))
	if response != expected {
		t.Fatalf("Expected %s, but got %s", expected, response)
	}

	h.deny = nil
	expected = "Unauthorized"
	response = strings.TrimSpace(testHTTPResponse(srv.URL, t))
	if response != expected {
		t.Fatalf("Expected %s, but got %s", expected, response)
	}

	addIPString(wl, "127.0.0.1", t)
	expected = "OK"
	response = strings.TrimSpace(testHTTPResponse(srv.URL, t))
	if response != expected {
		t.Fatalf("Expected %s, but got %s", expected, response)
	}
}

func TestFailHTTPFunc(t *testing.T) {
	wl := NewBasic()
	h, err := NewHandlerFunc(testAllowHandlerFunc, testDenyHandlerFunc, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	w := httptest.NewRecorder()
	req := new(http.Request)

	if h.ServeHTTP(w, req); w.Code != http.StatusInternalServerError {
		t.Fatalf("Expect HTTP 500, but got HTTP %d", w.Code)
	}
}

func TestBasicNetHTTP(t *testing.T) {
	wl := NewBasicNet()
	h, err := NewHandler(testAllowHandler, testDenyHandler, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	response := testHTTPResponse(srv.URL, t)
	if response != "NO" {
		t.Fatalf("Expected NO, but got %s", response)
	}

	testAddNet(wl, "127.0.0.1/32", t)
	response = testHTTPResponse(srv.URL, t)
	if response != "OK" {
		t.Fatalf("Expected OK, but got %s", response)
	}

	testDelNet(wl, "127.0.0.1/32", t)
	response = testHTTPResponse(srv.URL, t)
	if response != "NO" {
		t.Fatalf("Expected NO, but got %s", response)
	}
}

func TestBasicNetHTTPDefaultDeny(t *testing.T) {
	wl := NewBasicNet()
	h, err := NewHandler(testAllowHandler, nil, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	expected := "Unauthorized"
	response := strings.TrimSpace(testHTTPResponse(srv.URL, t))
	if response != expected {
		t.Fatalf("Expected %s, but got %s", expected, response)
	}
}

func TestBasicNetHTTPWorkers(t *testing.T) {
	wl := NewBasicNet()
	h, err := NewHandler(testAllowHandler, testDenyHandler, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	wg := new(sync.WaitGroup)
	defer srv.Close()

	for i := 0; i < 16; i++ {
		wg.Add(1)
		go testWorker(srv.URL, t, wg)
	}

	wg.Wait()

}

func TestNetFailHTTP(t *testing.T) {
	wl := NewBasicNet()
	h, err := NewHandler(testAllowHandler, testDenyHandler, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}
	w := httptest.NewRecorder()
	req := new(http.Request)

	if h.ServeHTTP(w, req); w.Code != http.StatusInternalServerError {
		t.Fatalf("Expect HTTP 500, but got HTTP %d", w.Code)
	}
}

func TestSetupNetHandlerFuncFails(t *testing.T) {
	wl := NewBasicNet()
	_, err := NewHandlerFunc(nil, testDenyHandlerFunc, wl)
	if err == nil {
		t.Fatal("expected NewHandlerFunc to fail with nil allow handler")
	}

	_, err = NewHandlerFunc(testAllowHandlerFunc, testDenyHandlerFunc, nil)
	if err == nil {
		t.Fatal("expected NewHandlerFunc to fail with nil whitelist")
	}

	_, err = NewHandlerFunc(testAllowHandlerFunc, nil, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}
}

func TestSetupNetHandlerFunc(t *testing.T) {
	wl := NewBasicNet()
	h, err := NewHandlerFunc(testAllowHandlerFunc, testDenyHandlerFunc, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	expected := "NO"
	response := strings.TrimSpace(testHTTPResponse(srv.URL, t))
	if response != expected {
		t.Fatalf("Expected %s, but got %s", expected, response)
	}

	h.deny = nil
	expected = "Unauthorized"
	response = strings.TrimSpace(testHTTPResponse(srv.URL, t))
	if response != expected {
		t.Fatalf("Expected %s, but got %s", expected, response)
	}

	testAddNet(wl, "127.0.0.1/32", t)
	expected = "OK"
	response = strings.TrimSpace(testHTTPResponse(srv.URL, t))
	if response != expected {
		t.Fatalf("Expected %s, but got %s", expected, response)
	}
}

func TestNetFailHTTPFunc(t *testing.T) {
	wl := NewBasicNet()
	h, err := NewHandlerFunc(testAllowHandlerFunc, testDenyHandlerFunc, wl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	w := httptest.NewRecorder()
	req := new(http.Request)

	if h.ServeHTTP(w, req); w.Code != http.StatusInternalServerError {
		t.Fatalf("Expect HTTP 500, but got HTTP %d", w.Code)
	}
}

func TestHandlerFunc(t *testing.T) {
	var acl ACL
	_, err := NewHandler(testAllowHandler, testDenyHandler, acl)
	if err == nil || err.Error() != "whitelist: ACL cannot be nil" {
		t.Fatal("Expected error with nil allow handler.")
	}

	acl = NewBasic()
	_, err = NewHandler(nil, testDenyHandler, acl)
	if err == nil || err.Error() != "whitelist: allow cannot be nil" {
		t.Fatal("Expected error with nil ACL.")
	}
}
