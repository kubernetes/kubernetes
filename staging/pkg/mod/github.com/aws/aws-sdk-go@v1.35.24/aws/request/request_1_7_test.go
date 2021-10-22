// +build !go1.8

package request

import (
	"net/http"
	"strings"
	"testing"
)

func TestResetBody_WithEmptyBody(t *testing.T) {
	r := Request{
		HTTPRequest: &http.Request{},
	}

	reader := strings.NewReader("")
	r.Body = reader

	r.ResetBody()

	if a, e := r.HTTPRequest.Body, (noBody{}); a != e {
		t.Errorf("expected request body to be set to reader, got %#v", r.HTTPRequest.Body)
	}
}
