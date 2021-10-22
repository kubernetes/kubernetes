package v2

import (
	"bytes"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
)

type signerBuilder struct {
	ServiceName  string
	Region       string
	SignTime     time.Time
	Query        url.Values
	Method       string
	SessionToken string
}

func (sb signerBuilder) BuildSigner() signer {
	endpoint := "https://" + sb.ServiceName + "." + sb.Region + ".amazonaws.com"
	var req *http.Request
	if sb.Method == "POST" {
		body := []byte(sb.Query.Encode())
		reader := bytes.NewReader(body)
		req, _ = http.NewRequest(sb.Method, endpoint, reader)
		req.Header.Add("Content-Type", "application/x-www-form-urlencoded")
		req.Header.Add("Content-Length", strconv.Itoa(len(body)))
	} else {
		req, _ = http.NewRequest(sb.Method, endpoint, nil)
		req.URL.RawQuery = sb.Query.Encode()
	}

	sig := signer{
		Request: req,
		Time:    sb.SignTime,
		Credentials: credentials.NewStaticCredentials(
			"AKID",
			"SECRET",
			sb.SessionToken),
	}

	if os.Getenv("DEBUG") != "" {
		sig.Debug = aws.LogDebug
		sig.Logger = aws.NewDefaultLogger()
	}

	return sig
}

func TestSignRequestWithAndWithoutSession(t *testing.T) {
	// have to create more than once, so use a function
	newQuery := func() url.Values {
		query := make(url.Values)
		query.Add("Action", "CreateDomain")
		query.Add("DomainName", "TestDomain-1437033376")
		query.Add("Version", "2009-04-15")
		return query
	}

	// create request without a SecurityToken (session) in the credentials

	query := newQuery()
	timestamp := time.Date(2015, 7, 16, 7, 56, 16, 0, time.UTC)
	builder := signerBuilder{
		Method:      "POST",
		ServiceName: "sdb",
		Region:      "ap-southeast-2",
		SignTime:    timestamp,
		Query:       query,
	}

	signer := builder.BuildSigner()

	err := signer.Sign()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := "tm4dX8Ks7pzFSVHz7qHdoJVXKRLuC4gWz9eti60d8ks=", signer.signature; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := 8, len(signer.Query); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "AKID", signer.Query.Get("AWSAccessKeyId"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "2015-07-16T07:56:16Z", signer.Query.Get("Timestamp"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "HmacSHA256", signer.Query.Get("SignatureMethod"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "2", signer.Query.Get("SignatureVersion"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "tm4dX8Ks7pzFSVHz7qHdoJVXKRLuC4gWz9eti60d8ks=", signer.Query.Get("Signature"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "CreateDomain", signer.Query.Get("Action"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "TestDomain-1437033376", signer.Query.Get("DomainName"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "2009-04-15", signer.Query.Get("Version"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	// should not have a SecurityToken parameter
	_, ok := signer.Query["SecurityToken"]
	if ok {
		t.Errorf("expect SecurityToken found, was not")
	}

	// now sign again, this time with a security token (session)

	query = newQuery()
	builder.SessionToken = "SESSION"
	signer = builder.BuildSigner()

	err = signer.Sign()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := "Ch6qv3rzXB1SLqY2vFhsgA1WQ9rnQIE2WJCigOvAJwI=", signer.signature; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := 9, len(signer.Query); e != a { // expect one more parameter
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "Ch6qv3rzXB1SLqY2vFhsgA1WQ9rnQIE2WJCigOvAJwI=", signer.Query.Get("Signature"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "SESSION", signer.Query.Get("SecurityToken"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestMoreComplexSignRequest(t *testing.T) {
	query := make(url.Values)
	query.Add("Action", "PutAttributes")
	query.Add("DomainName", "TestDomain-1437041569")
	query.Add("Version", "2009-04-15")
	query.Add("Attribute.2.Name", "Attr2")
	query.Add("Attribute.2.Value", "Value2")
	query.Add("Attribute.2.Replace", "true")
	query.Add("Attribute.1.Name", "Attr1-%\\+ %")
	query.Add("Attribute.1.Value", " \tValue1 +!@#$%^&*(){}[]\"';:?/.>,<\x12\x00")
	query.Add("Attribute.1.Replace", "true")
	query.Add("ItemName", "Item 1")

	timestamp := time.Date(2015, 7, 16, 10, 12, 51, 0, time.UTC)
	builder := signerBuilder{
		Method:       "POST",
		ServiceName:  "sdb",
		Region:       "ap-southeast-2",
		SignTime:     timestamp,
		Query:        query,
		SessionToken: "SESSION",
	}

	signer := builder.BuildSigner()

	err := signer.Sign()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := "WNdE62UJKLKoA6XncVY/9RDbrKmcVMdQPQOTAs8SgwQ=", signer.signature; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestGet(t *testing.T) {
	svc := awstesting.NewClient(&aws.Config{
		Credentials: credentials.NewStaticCredentials("AKID", "SECRET", "SESSION"),
		Region:      aws.String("ap-southeast-2"),
	})
	r := svc.NewRequest(
		&request.Operation{
			Name:       "OpName",
			HTTPMethod: "GET",
			HTTPPath:   "/",
		},
		nil,
		nil,
	)

	r.Build()
	if e, a := "GET", r.HTTPRequest.Method; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "", r.HTTPRequest.URL.Query().Get("Signature"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	SignSDKRequest(r)
	if r.Error != nil {
		t.Fatalf("expect no error, got %v", r.Error)
	}
	t.Logf("Signature: %s", r.HTTPRequest.URL.Query().Get("Signature"))
	if len(r.HTTPRequest.URL.Query().Get("Signature")) == 0 {
		t.Errorf("expect signature to be set, was not")
	}
}

func TestAnonymousCredentials(t *testing.T) {
	svc := awstesting.NewClient(&aws.Config{
		Credentials: credentials.AnonymousCredentials,
		Region:      aws.String("ap-southeast-2"),
	})
	r := svc.NewRequest(
		&request.Operation{
			Name:       "PutAttributes",
			HTTPMethod: "POST",
			HTTPPath:   "/",
		},
		nil,
		nil,
	)
	r.Build()

	SignSDKRequest(r)

	req := r.HTTPRequest
	req.ParseForm()

	if a := req.PostForm.Get("Signature"); len(a) != 0 {
		t.Errorf("expect no signature, got %v", a)
	}
}
