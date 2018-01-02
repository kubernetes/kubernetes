package httputil

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestDeleteCookies(t *testing.T) {
	tests := []struct {
		// cookie names to delete
		n []string
	}{
		// single
		{
			n: []string{"foo"},
		},
		// multiple
		{
			n: []string{"foo", "bar"},
		},
	}

	for i, tt := range tests {
		w := httptest.NewRecorder()
		DeleteCookies(w, tt.n...)
		resp := &http.Response{}
		resp.Header = w.Header()
		cks := resp.Cookies()

		if len(cks) != len(tt.n) {
			t.Errorf("case %d: unexpected number of cookies, want: %d, got: %d", i, len(tt.n), len(cks))
		}

		for _, c := range cks {
			if c.Value != "" {
				t.Errorf("case %d: unexpected cookie value, want: %q, got: %q", i, "", c.Value)
			}
			if c.Path != "/" {
				t.Errorf("case %d: unexpected cookie path, want: %q, got: %q", i, "/", c.Path)
			}
			if c.MaxAge != -1 {
				t.Errorf("case %d: unexpected cookie max-age, want: %q, got: %q", i, -1, c.MaxAge)
			}
			if !c.Expires.IsZero() {
				t.Errorf("case %d: unexpected cookie expires, want: %q, got: %q", i, time.Time{}, c.MaxAge)
			}
		}
	}
}
