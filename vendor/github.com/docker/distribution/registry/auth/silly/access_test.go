package silly

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/auth"
)

func TestSillyAccessController(t *testing.T) {
	ac := &accessController{
		realm:   "test-realm",
		service: "test-service",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := context.WithRequest(context.Background(), r)
		authCtx, err := ac.Authorized(ctx)
		if err != nil {
			switch err := err.(type) {
			case auth.Challenge:
				err.SetHeaders(w)
				w.WriteHeader(http.StatusUnauthorized)
				return
			default:
				t.Fatalf("unexpected error authorizing request: %v", err)
			}
		}

		userInfo, ok := authCtx.Value(auth.UserKey).(auth.UserInfo)
		if !ok {
			t.Fatal("silly accessController did not set auth.user context")
		}

		if userInfo.Name != "silly" {
			t.Fatalf("expected user name %q, got %q", "silly", userInfo.Name)
		}

		w.WriteHeader(http.StatusNoContent)
	}))

	resp, err := http.Get(server.URL)
	if err != nil {
		t.Fatalf("unexpected error during GET: %v", err)
	}
	defer resp.Body.Close()

	// Request should not be authorized
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unexpected response status: %v != %v", resp.StatusCode, http.StatusUnauthorized)
	}

	req, err := http.NewRequest("GET", server.URL, nil)
	if err != nil {
		t.Fatalf("unexpected error creating new request: %v", err)
	}
	req.Header.Set("Authorization", "seriously, anything")

	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("unexpected error during GET: %v", err)
	}
	defer resp.Body.Close()

	// Request should not be authorized
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("unexpected response status: %v != %v", resp.StatusCode, http.StatusNoContent)
	}
}
