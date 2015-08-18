package serve

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	rice "github.com/GeertJohan/go.rice"
)

func TestServe(t *testing.T) {
	registerHandlers()
	ts := httptest.NewServer(http.DefaultServeMux)
	defer ts.Close()
	expected := make(map[string]int)
	for endpoint := range v1Endpoints {
		expected[v1APIPath(endpoint)] = http.StatusOK
	}

	staticDir := "static"
	err := rice.MustFindBox(staticDir).Walk("", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() {
			expected["/"+path] = http.StatusOK
		}
		return nil
	})
	if err != nil {
		t.Error(err)
	}

	// Disabled endpoints should return '404 Not Found'
	expected[v1APIPath("sign")] = http.StatusNotFound
	expected[v1APIPath("authsign")] = http.StatusNotFound
	expected[v1APIPath("newcert")] = http.StatusNotFound
	expected[v1APIPath("info")] = http.StatusNotFound
	expected[v1APIPath("bundle")] = http.StatusNotFound

	// Enabled endpoints should return '405 Method Not Allowed'
	expected[v1APIPath("init_ca")] = http.StatusMethodNotAllowed
	expected[v1APIPath("newkey")] = http.StatusMethodNotAllowed

	// POST-only endpoints should return '400 Bad Request'
	expected[v1APIPath("scan")] = http.StatusBadRequest

	// Non-existent endpoints should return '404 Not Found'
	expected["/bad_endpoint"] = http.StatusNotFound

	for endpoint, status := range expected {
		resp, err := http.Get(ts.URL + endpoint)
		if err != nil {
			t.Error(err)
		}
		if resp.StatusCode != status {
			t.Fatalf("%s: '%s' (expected '%s')", endpoint, resp.Status, http.StatusText(status))
		}
	}
}
