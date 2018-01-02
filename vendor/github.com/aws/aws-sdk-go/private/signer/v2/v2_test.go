package v2

import (
	"bytes"
	"net/http"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/stretchr/testify/assert"
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
		req.Header.Add("Content-Length", string(len(body)))
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
	assert := assert.New(t)

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
	assert.NoError(err)
	assert.Equal("tm4dX8Ks7pzFSVHz7qHdoJVXKRLuC4gWz9eti60d8ks=", signer.signature)
	assert.Equal(8, len(signer.Query))
	assert.Equal("AKID", signer.Query.Get("AWSAccessKeyId"))
	assert.Equal("2015-07-16T07:56:16Z", signer.Query.Get("Timestamp"))
	assert.Equal("HmacSHA256", signer.Query.Get("SignatureMethod"))
	assert.Equal("2", signer.Query.Get("SignatureVersion"))
	assert.Equal("tm4dX8Ks7pzFSVHz7qHdoJVXKRLuC4gWz9eti60d8ks=", signer.Query.Get("Signature"))
	assert.Equal("CreateDomain", signer.Query.Get("Action"))
	assert.Equal("TestDomain-1437033376", signer.Query.Get("DomainName"))
	assert.Equal("2009-04-15", signer.Query.Get("Version"))

	// should not have a SecurityToken parameter
	_, ok := signer.Query["SecurityToken"]
	assert.False(ok)

	// now sign again, this time with a security token (session)

	query = newQuery()
	builder.SessionToken = "SESSION"
	signer = builder.BuildSigner()

	err = signer.Sign()
	assert.NoError(err)
	assert.Equal("Ch6qv3rzXB1SLqY2vFhsgA1WQ9rnQIE2WJCigOvAJwI=", signer.signature)
	assert.Equal(9, len(signer.Query)) // expect one more parameter
	assert.Equal("Ch6qv3rzXB1SLqY2vFhsgA1WQ9rnQIE2WJCigOvAJwI=", signer.Query.Get("Signature"))
	assert.Equal("SESSION", signer.Query.Get("SecurityToken"))
}

func TestMoreComplexSignRequest(t *testing.T) {
	assert := assert.New(t)
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
	assert.NoError(err)
	assert.Equal("WNdE62UJKLKoA6XncVY/9RDbrKmcVMdQPQOTAs8SgwQ=", signer.signature)
}

func TestGet(t *testing.T) {
	assert := assert.New(t)
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
	assert.Equal("GET", r.HTTPRequest.Method)
	assert.Equal("", r.HTTPRequest.URL.Query().Get("Signature"))

	SignSDKRequest(r)
	assert.NoError(r.Error)
	t.Logf("Signature: %s", r.HTTPRequest.URL.Query().Get("Signature"))
	assert.NotEqual("", r.HTTPRequest.URL.Query().Get("Signature"))
}

func TestAnonymousCredentials(t *testing.T) {
	assert := assert.New(t)
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

	assert.Empty(req.PostForm.Get("Signature"))
}
