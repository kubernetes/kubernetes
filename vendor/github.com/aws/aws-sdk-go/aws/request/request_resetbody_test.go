package request

import (
	"bytes"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
)

func TestResetBody_WithBodyContents(t *testing.T) {
	r := Request{
		HTTPRequest: &http.Request{},
	}

	reader := strings.NewReader("abc")
	r.Body = reader

	r.ResetBody()

	if v, ok := r.HTTPRequest.Body.(*offsetReader); !ok || v == nil {
		t.Errorf("expected request body to be set to reader, got %#v",
			r.HTTPRequest.Body)
	}
}

func TestResetBody_ExcludeUnseekableBodyByMethod(t *testing.T) {
	cases := []struct {
		Method   string
		IsNoBody bool
	}{
		{"GET", true},
		{"HEAD", true},
		{"DELETE", true},
		{"PUT", false},
		{"PATCH", false},
		{"POST", false},
	}

	reader := aws.ReadSeekCloser(bytes.NewBuffer([]byte("abc")))

	for i, c := range cases {
		r := Request{
			HTTPRequest: &http.Request{},
			Operation: &Operation{
				HTTPMethod: c.Method,
			},
		}

		r.SetReaderBody(reader)

		if a, e := r.HTTPRequest.Body == NoBody, c.IsNoBody; a != e {
			t.Errorf("%d, expect body to be set to noBody(%t), but was %t", i, e, a)
		}
	}

}
