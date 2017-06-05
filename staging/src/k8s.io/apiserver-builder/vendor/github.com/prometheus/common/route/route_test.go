package route

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRedirect(t *testing.T) {
	router := New().WithPrefix("/test/prefix")
	w := httptest.NewRecorder()
	r, err := http.NewRequest("GET", "http://localhost:9090/foo", nil)
	if err != nil {
		t.Fatalf("Error building test request: %s", err)
	}

	router.Redirect(w, r, "/some/endpoint", http.StatusFound)
	if w.Code != http.StatusFound {
		t.Fatalf("Unexpected redirect status code: got %d, want %d", w.Code, http.StatusFound)
	}

	want := "/test/prefix/some/endpoint"
	got := w.Header()["Location"][0]
	if want != got {
		t.Fatalf("Unexpected redirect location: got %s, want %s", got, want)
	}
}
