package s3_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

const (
	metaKeyPrefix = `X-Amz-Meta-`
	utf8KeySuffix = `My-Info`
	utf8Value     = "hello-世界\u0444"
)

func TestPutObjectMetadataWithUnicode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if e, a := utf8Value, r.Header.Get(metaKeyPrefix+utf8KeySuffix); e != a {
			t.Errorf("expected %s, but received %s", e, a)
		}
	}))
	defer server.Close()

	svc := s3.New(unit.Session, &aws.Config{
		Endpoint:   aws.String(server.URL),
		DisableSSL: aws.Bool(true),
	})

	_, err := svc.PutObject(&s3.PutObjectInput{
		Bucket: aws.String("my_bucket"),
		Key:    aws.String("my_key"),
		Body:   strings.NewReader(""),
		Metadata: func() map[string]*string {
			v := map[string]*string{}
			v[utf8KeySuffix] = aws.String(utf8Value)
			return v
		}(),
	})

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
}

func TestGetObjectMetadataWithUnicode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(metaKeyPrefix+utf8KeySuffix, utf8Value)
	}))
	defer server.Close()

	svc := s3.New(unit.Session, &aws.Config{
		Endpoint:   aws.String(server.URL),
		DisableSSL: aws.Bool(true),
	})

	resp, err := svc.GetObject(&s3.GetObjectInput{
		Bucket: aws.String("my_bucket"),
		Key:    aws.String("my_key"),
	})

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	resp.Body.Close()

	if e, a := utf8Value, *resp.Metadata[utf8KeySuffix]; e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}

}
