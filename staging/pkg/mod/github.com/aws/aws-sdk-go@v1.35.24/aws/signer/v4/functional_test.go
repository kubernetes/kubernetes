package v4_test

import (
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
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

func epochTime() time.Time { return time.Unix(0, 0) }

func TestPresignHandler(t *testing.T) {
	svc := s3.New(unit.Session)
	svc.Handlers.Sign.SwapNamed(request.NamedHandler{
		Name: v4.SignRequestHandler.Name,
		Fn: func(r *request.Request) {
			v4.SignSDKRequestWithCurrentTime(r, epochTime)
		},
	})

	req, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket:             aws.String("bucket"),
		Key:                aws.String("key"),
		ContentDisposition: aws.String("a+b c$d"),
		ACL:                aws.String("public-read"),
	})
	req.Time = epochTime()
	urlstr, err := req.Presign(5 * time.Minute)

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	expectedHost := "bucket.s3.mock-region.amazonaws.com"
	expectedDate := "19700101T000000Z"
	expectedHeaders := "content-disposition;host;x-amz-acl"
	expectedSig := "2d76a414208c0eac2a23ef9c834db9635ecd5a0fbb447a00ad191f82d854f55b"
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
	if a := urlQ.Get("X-Amz-Content-Sha256"); len(a) != 0 {
		t.Errorf("expect no content sha256 got %v", a)
	}

	if e, a := "+", urlstr; strings.Contains(a, e) { // + encoded as %20
		t.Errorf("expect %v not to be in %v", e, a)
	}
}

func TestPresignRequest(t *testing.T) {
	svc := s3.New(unit.Session)
	svc.Handlers.Sign.SwapNamed(request.NamedHandler{
		Name: v4.SignRequestHandler.Name,
		Fn: func(r *request.Request) {
			v4.SignSDKRequestWithCurrentTime(r, epochTime)
		},
	})

	req, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket:             aws.String("bucket"),
		Key:                aws.String("key"),
		ContentDisposition: aws.String("a+b c$d"),
		ACL:                aws.String("public-read"),
	})
	req.Time = epochTime()
	urlstr, headers, err := req.PresignRequest(5 * time.Minute)

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	expectedHost := "bucket.s3.mock-region.amazonaws.com"
	expectedDate := "19700101T000000Z"
	expectedHeaders := "content-disposition;host;x-amz-acl"
	expectedSig := "2d76a414208c0eac2a23ef9c834db9635ecd5a0fbb447a00ad191f82d854f55b"
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
	if a := urlQ.Get("X-Amz-Content-Sha256"); len(a) != 0 {
		t.Errorf("expect no content sha256 got %v", a)
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

	_, err = signer.Sign(req, nil, "es", "us-east-1", epochTime())
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	actual := req.Header.Get("Authorization")
	if e, a := expectSig, actual; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestStandaloneSign_WithPort(t *testing.T) {

	cases := []struct {
		description string
		url         string
		expectedSig string
	}{
		{
			"default HTTPS port",
			"https://estest.us-east-1.es.amazonaws.com:443/_search",
			"AWS4-HMAC-SHA256 Credential=AKID/19700101/us-east-1/es/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, Signature=e573fc9aa3a156b720976419319be98fb2824a3abc2ddd895ecb1d1611c6a82d",
		},
		{
			"default HTTP port",
			"http://example.com:80/_search",
			"AWS4-HMAC-SHA256 Credential=AKID/19700101/us-east-1/es/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, Signature=54ebe60c4ae03a40948b849e13c333523235f38002e2807059c64a9a8c7cb951",
		},
		{
			"non-standard HTTP port",
			"http://example.com:9200/_search",
			"AWS4-HMAC-SHA256 Credential=AKID/19700101/us-east-1/es/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, Signature=cd9d926a460f8d3b58b57beadbd87666dc667e014c0afaa4cea37b2867f51b4f",
		},
		{
			"non-standard HTTPS port",
			"https://example.com:9200/_search",
			"AWS4-HMAC-SHA256 Credential=AKID/19700101/us-east-1/es/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, Signature=cd9d926a460f8d3b58b57beadbd87666dc667e014c0afaa4cea37b2867f51b4f",
		},
	}

	for _, c := range cases {
		signer := v4.NewSigner(unit.Session.Config.Credentials)
		req, _ := http.NewRequest("GET", c.url, nil)
		_, err := signer.Sign(req, nil, "es", "us-east-1", epochTime())
		if err != nil {
			t.Fatalf("expect no error, got %v", err)
		}

		actual := req.Header.Get("Authorization")
		if e, a := c.expectedSig, actual; e != a {
			t.Errorf("%s, expect %v, got %v", c.description, e, a)
		}
	}
}

func TestStandalonePresign_WithPort(t *testing.T) {

	cases := []struct {
		description string
		url         string
		expectedSig string
	}{
		{
			"default HTTPS port",
			"https://estest.us-east-1.es.amazonaws.com:443/_search",
			"0abcf61a351063441296febf4b485734d780634fba8cf1e7d9769315c35255d6",
		},
		{
			"default HTTP port",
			"http://example.com:80/_search",
			"fce9976dd6c849c21adfa6d3f3e9eefc651d0e4a2ccd740d43efddcccfdc8179",
		},
		{
			"non-standard HTTP port",
			"http://example.com:9200/_search",
			"f33c25a81c735e42bef35ed5e9f720c43940562e3e616ff0777bf6dde75249b0",
		},
		{
			"non-standard HTTPS port",
			"https://example.com:9200/_search",
			"f33c25a81c735e42bef35ed5e9f720c43940562e3e616ff0777bf6dde75249b0",
		},
	}

	for _, c := range cases {
		signer := v4.NewSigner(unit.Session.Config.Credentials)
		req, _ := http.NewRequest("GET", c.url, nil)
		_, err := signer.Presign(req, nil, "es", "us-east-1", 5*time.Minute, epochTime())
		if err != nil {
			t.Fatalf("expect no error, got %v", err)
		}

		actual := req.URL.Query().Get("X-Amz-Signature")
		if e, a := c.expectedSig, actual; e != a {
			t.Errorf("%s, expect %v, got %v", c.description, e, a)
		}
	}
}
