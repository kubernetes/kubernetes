package registry

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestEndpointParse(t *testing.T) {
	testData := []struct {
		str      string
		expected string
	}{
		{INDEXSERVER, INDEXSERVER},
		{"http://0.0.0.0:5000/v1/", "http://0.0.0.0:5000/v1/"},
		{"http://0.0.0.0:5000/v2/", "http://0.0.0.0:5000/v2/"},
		{"http://0.0.0.0:5000", "http://0.0.0.0:5000/v0/"},
		{"0.0.0.0:5000", "https://0.0.0.0:5000/v0/"},
	}
	for _, td := range testData {
		e, err := newEndpoint(td.str, nil, nil)
		if err != nil {
			t.Errorf("%q: %s", td.str, err)
		}
		if e == nil {
			t.Logf("something's fishy, endpoint for %q is nil", td.str)
			continue
		}
		if e.String() != td.expected {
			t.Errorf("expected %q, got %q", td.expected, e.String())
		}
	}
}

// Ensure that a registry endpoint that responds with a 401 only is determined
// to be a v1 registry unless it includes a valid v2 API header.
func TestValidateEndpointAmbiguousAPIVersion(t *testing.T) {
	requireBasicAuthHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("WWW-Authenticate", `Basic realm="localhost"`)
		w.WriteHeader(http.StatusUnauthorized)
	})

	requireBasicAuthHandlerV2 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// This mock server supports v2.0, v2.1, v42.0, and v100.0
		w.Header().Add("Docker-Distribution-API-Version", "registry/100.0 registry/42.0")
		w.Header().Add("Docker-Distribution-API-Version", "registry/2.0 registry/2.1")
		requireBasicAuthHandler.ServeHTTP(w, r)
	})

	// Make a test server which should validate as a v1 server.
	testServer := httptest.NewServer(requireBasicAuthHandler)
	defer testServer.Close()

	testServerURL, err := url.Parse(testServer.URL)
	if err != nil {
		t.Fatal(err)
	}

	testEndpoint := Endpoint{
		URL:     testServerURL,
		Version: APIVersionUnknown,
		client:  HTTPClient(NewTransport(nil)),
	}

	if err = validateEndpoint(&testEndpoint); err != nil {
		t.Fatal(err)
	}

	if testEndpoint.Version != APIVersion1 {
		t.Fatalf("expected endpoint to validate to %d, got %d", APIVersion1, testEndpoint.Version)
	}

	// Make a test server which should validate as a v2 server.
	testServer = httptest.NewServer(requireBasicAuthHandlerV2)
	defer testServer.Close()

	testServerURL, err = url.Parse(testServer.URL)
	if err != nil {
		t.Fatal(err)
	}

	testEndpoint.URL = testServerURL
	testEndpoint.Version = APIVersionUnknown

	if err = validateEndpoint(&testEndpoint); err != nil {
		t.Fatal(err)
	}

	if testEndpoint.Version != APIVersion2 {
		t.Fatalf("expected endpoint to validate to %d, got %d", APIVersion2, testEndpoint.Version)
	}
}
