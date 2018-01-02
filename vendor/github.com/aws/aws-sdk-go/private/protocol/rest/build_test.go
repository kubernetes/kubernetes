package rest

import (
	"net/http"
	"net/url"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
)

func TestCleanPath(t *testing.T) {
	uri := &url.URL{
		Path:   "//foo//bar",
		Scheme: "https",
		Host:   "host",
	}
	cleanPath(uri)

	expected := "https://host/foo/bar"
	if a, e := uri.String(), expected; a != e {
		t.Errorf("expect %q URI, got %q", e, a)
	}
}

func TestMarshalPath(t *testing.T) {
	in := struct {
		Bucket *string `location:"uri" locationName:"bucket"`
		Key    *string `location:"uri" locationName:"key"`
	}{
		Bucket: aws.String("mybucket"),
		Key:    aws.String("my/cool+thing space/object世界"),
	}

	expectURL := `/mybucket/my/cool+thing space/object世界`
	expectEscapedURL := `/mybucket/my/cool%2Bthing%20space/object%E4%B8%96%E7%95%8C`

	req := &request.Request{
		HTTPRequest: &http.Request{
			URL: &url.URL{Scheme: "https", Host: "exmaple.com", Path: "/{bucket}/{key+}"},
		},
		Params: &in,
	}

	Build(req)

	if req.Error != nil {
		t.Fatalf("unexpected error, %v", req.Error)
	}

	if a, e := req.HTTPRequest.URL.Path, expectURL; a != e {
		t.Errorf("expect %q URI, got %q", e, a)
	}

	if a, e := req.HTTPRequest.URL.RawPath, expectEscapedURL; a != e {
		t.Errorf("expect %q escaped URI, got %q", e, a)
	}

	if a, e := req.HTTPRequest.URL.EscapedPath(), expectEscapedURL; a != e {
		t.Errorf("expect %q escaped URI, got %q", e, a)
	}

}
