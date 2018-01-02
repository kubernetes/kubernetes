package v4_test

import (
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/signer/v4"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

var standaloneSignCases = []struct {
	OrigURI                    string
	OrigQuery                  string
	Region, Service, SubDomain string
	ExpSig                     string
	EscapedURI                 string
}{
	{
		OrigURI:   `/logs-*/_search`,
		OrigQuery: `pretty=true`,
		Region:    "us-west-2", Service: "es", SubDomain: "hostname-clusterkey",
		EscapedURI: `/logs-%2A/_search`,
		ExpSig:     `AWS4-HMAC-SHA256 Credential=AKID/19700101/us-west-2/es/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, Signature=79d0760751907af16f64a537c1242416dacf51204a7dd5284492d15577973b91`,
	},
}

func TestPresignHandler(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket:             aws.String("bucket"),
		Key:                aws.String("key"),
		ContentDisposition: aws.String("a+b c$d"),
		ACL:                aws.String("public-read"),
	})
	req.Time = time.Unix(0, 0)
	urlstr, err := req.Presign(5 * time.Minute)

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	expectedHost := "bucket.s3.mock-region.amazonaws.com"
	expectedDate := "19700101T000000Z"
	expectedHeaders := "content-disposition;host;x-amz-acl"
	expectedSig := "a46583256431b09eb45ba4af2e6286d96a9835ed13721023dc8076dfdcb90fcb"
	expectedCred := "AKID/19700101/mock-region/s3/aws4_request"

	u, _ := url.Parse(urlstr)
	urlQ := u.Query()
	if e, a := expectedHost, u.Host; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedSig, urlQ.Get("X-Amz-Signature"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedCred, urlQ.Get("X-Amz-Credential"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedHeaders, urlQ.Get("X-Amz-SignedHeaders"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedDate, urlQ.Get("X-Amz-Date"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "300", urlQ.Get("X-Amz-Expires"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "UNSIGNED-PAYLOAD", urlQ.Get("X-Amz-Content-Sha256"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	if e, a := "+", urlstr; strings.Contains(a, e) { // + encoded as %20
		t.Errorf("expect %v not to be in %v", e, a)
	}
}

func TestPresignRequest(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket:             aws.String("bucket"),
		Key:                aws.String("key"),
		ContentDisposition: aws.String("a+b c$d"),
		ACL:                aws.String("public-read"),
	})
	req.Time = time.Unix(0, 0)
	urlstr, headers, err := req.PresignRequest(5 * time.Minute)

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	expectedHost := "bucket.s3.mock-region.amazonaws.com"
	expectedDate := "19700101T000000Z"
	expectedHeaders := "content-disposition;host;x-amz-acl"
	expectedSig := "a46583256431b09eb45ba4af2e6286d96a9835ed13721023dc8076dfdcb90fcb"
	expectedCred := "AKID/19700101/mock-region/s3/aws4_request"
	expectedHeaderMap := http.Header{
		"x-amz-acl":           []string{"public-read"},
		"content-disposition": []string{"a+b c$d"},
	}

	u, _ := url.Parse(urlstr)
	urlQ := u.Query()
	if e, a := expectedHost, u.Host; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedSig, urlQ.Get("X-Amz-Signature"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedCred, urlQ.Get("X-Amz-Credential"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedHeaders, urlQ.Get("X-Amz-SignedHeaders"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedDate, urlQ.Get("X-Amz-Date"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedHeaderMap, headers; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "300", urlQ.Get("X-Amz-Expires"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "UNSIGNED-PAYLOAD", urlQ.Get("X-Amz-Content-Sha256"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	if e, a := "+", urlstr; strings.Contains(a, e) { // + encoded as %20
		t.Errorf("expect %v not to be in %v", e, a)
	}
}

func TestStandaloneSign_CustomURIEscape(t *testing.T) {
	var expectSig = `AWS4-HMAC-SHA256 Credential=AKID/19700101/us-east-1/es/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, Signature=6601e883cc6d23871fd6c2a394c5677ea2b8c82b04a6446786d64cd74f520967`

	creds := unit.Session.Config.Credentials
	signer := v4.NewSigner(creds, func(s *v4.Signer) {
		s.DisableURIPathEscaping = true
	})

	host := "https://subdomain.us-east-1.es.amazonaws.com"
	req, err := http.NewRequest("GET", host, nil)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	req.URL.Path = `/log-*/_search`
	req.URL.Opaque = "//subdomain.us-east-1.es.amazonaws.com/log-%2A/_search"

	_, err = signer.Sign(req, nil, "es", "us-east-1", time.Unix(0, 0))
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	actual := req.Header.Get("Authorization")
	if e, a := expectSig, actual; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}
