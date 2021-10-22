package ec2rolecreds_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

const credsRespTmpl = `{
  "Code": "Success",
  "Type": "AWS-HMAC",
  "AccessKeyId" : "accessKey",
  "SecretAccessKey" : "secret",
  "Token" : "token",
  "Expiration" : "%s",
  "LastUpdated" : "2009-11-23T0:00:00Z"
}`

const credsFailRespTmpl = `{
  "Code": "ErrorCode",
  "Message": "ErrorMsg",
  "LastUpdated": "2009-11-23T0:00:00Z"
}`

func initTestServer(expireOn string, failAssume bool) *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/latest/meta-data/iam/security-credentials/" {
			fmt.Fprintln(w, "RoleName")
		} else if r.URL.Path == "/latest/meta-data/iam/security-credentials/RoleName" {
			if failAssume {
				fmt.Fprintf(w, credsFailRespTmpl)
			} else {
				fmt.Fprintf(w, credsRespTmpl, expireOn)
			}
		} else {
			http.Error(w, "Not found", http.StatusNotFound)
		}
	}))

	return server
}

func TestEC2RoleProvider(t *testing.T) {
	server := initTestServer("2014-12-16T01:51:37Z", false)
	defer server.Close()

	p := &ec2rolecreds.EC2RoleProvider{
		Client: ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")}),
	}

	creds, err := p.Retrieve()
	if err != nil {
		t.Errorf("Expect no error, got %v", err)
	}

	if e, a := "accessKey", creds.AccessKeyID; e != a {
		t.Errorf("Expect access key ID to match, %v got %v", e, a)
	}
	if e, a := "secret", creds.SecretAccessKey; e != a {
		t.Errorf("Expect secret access key to match, %v got %v", e, a)
	}
	if e, a := "token", creds.SessionToken; e != a {
		t.Errorf("Expect session token to match, %v got %v", e, a)
	}
}

func TestEC2RoleProviderFailAssume(t *testing.T) {
	server := initTestServer("2014-12-16T01:51:37Z", true)
	defer server.Close()

	p := &ec2rolecreds.EC2RoleProvider{
		Client: ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")}),
	}

	creds, err := p.Retrieve()
	if err == nil {
		t.Errorf("Expect error")
	}

	e := err.(awserr.Error)
	if e, a := "ErrorCode", e.Code(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "ErrorMsg", e.Message(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if v := e.OrigErr(); v != nil {
		t.Errorf("expect nil, got %v", v)
	}

	if e, a := "", creds.AccessKeyID; e != a {
		t.Errorf("Expect access key ID to match, %v got %v", e, a)
	}
	if e, a := "", creds.SecretAccessKey; e != a {
		t.Errorf("Expect secret access key to match, %v got %v", e, a)
	}
	if e, a := "", creds.SessionToken; e != a {
		t.Errorf("Expect session token to match, %v got %v", e, a)
	}
}

func TestEC2RoleProviderIsExpired(t *testing.T) {
	server := initTestServer("2014-12-16T01:51:37Z", false)
	defer server.Close()

	p := &ec2rolecreds.EC2RoleProvider{
		Client: ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")}),
	}
	p.CurrentTime = func() time.Time {
		return time.Date(2014, 12, 15, 21, 26, 0, 0, time.UTC)
	}

	if !p.IsExpired() {
		t.Errorf("Expect creds to be expired before retrieve.")
	}

	_, err := p.Retrieve()
	if v := err; v != nil {
		t.Errorf("Expect no error, %v", err)
	}

	if p.IsExpired() {
		t.Errorf("Expect creds to not be expired after retrieve.")
	}

	p.CurrentTime = func() time.Time {
		return time.Date(3014, 12, 15, 21, 26, 0, 0, time.UTC)
	}

	if !p.IsExpired() {
		t.Errorf("Expect creds to be expired.")
	}
}

func TestEC2RoleProviderExpiryWindowIsExpired(t *testing.T) {
	server := initTestServer("2014-12-16T01:51:37Z", false)
	defer server.Close()

	p := &ec2rolecreds.EC2RoleProvider{
		Client:       ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")}),
		ExpiryWindow: time.Hour * 1,
	}
	p.CurrentTime = func() time.Time {
		return time.Date(2014, 12, 15, 0, 51, 37, 0, time.UTC)
	}

	if !p.IsExpired() {
		t.Errorf("Expect creds to be expired before retrieve.")
	}

	_, err := p.Retrieve()
	if v := err; v != nil {
		t.Errorf("Expect no error, %v", err)
	}

	if p.IsExpired() {
		t.Errorf("Expect creds to not be expired after retrieve.")
	}

	p.CurrentTime = func() time.Time {
		return time.Date(2014, 12, 16, 0, 55, 37, 0, time.UTC)
	}

	if !p.IsExpired() {
		t.Errorf("Expect creds to be expired.")
	}
}

func BenchmarkEC3RoleProvider(b *testing.B) {
	server := initTestServer("2014-12-16T01:51:37Z", false)
	defer server.Close()

	p := &ec2rolecreds.EC2RoleProvider{
		Client: ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")}),
	}
	_, err := p.Retrieve()
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.Retrieve(); err != nil {
			b.Fatal(err)
		}
	}
}
