package registry

import (
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
)

type handlerFunc func(w http.ResponseWriter, r *http.Request)

// Mock represent a registry mock
type Mock struct {
	server   *httptest.Server
	hostport string
	handlers map[string]handlerFunc
	mu       sync.Mutex
}

// RegisterHandler register the specified handler for the registry mock
func (tr *Mock) RegisterHandler(path string, h handlerFunc) {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	tr.handlers[path] = h
}

// NewMock creates a registry mock
func NewMock(t testingT) (*Mock, error) {
	testReg := &Mock{handlers: make(map[string]handlerFunc)}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		url := r.URL.String()

		var matched bool
		var err error
		for re, function := range testReg.handlers {
			matched, err = regexp.MatchString(re, url)
			if err != nil {
				t.Fatal("Error with handler regexp")
			}
			if matched {
				function(w, r)
				break
			}
		}

		if !matched {
			t.Fatalf("Unable to match %s with regexp", url)
		}
	}))

	testReg.server = ts
	testReg.hostport = strings.Replace(ts.URL, "http://", "", 1)
	return testReg, nil
}

// URL returns the url of the registry
func (tr *Mock) URL() string {
	return tr.hostport
}

// Close closes mock and releases resources
func (tr *Mock) Close() {
	tr.server.Close()
}
