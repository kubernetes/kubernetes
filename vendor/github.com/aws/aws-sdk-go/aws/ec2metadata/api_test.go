package ec2metadata_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"path"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

const instanceIdentityDocument = `{
  "devpayProductCodes" : null,
  "availabilityZone" : "us-east-1d",
  "privateIp" : "10.158.112.84",
  "version" : "2010-08-31",
  "region" : "us-east-1",
  "instanceId" : "i-1234567890abcdef0",
  "billingProducts" : null,
  "instanceType" : "t1.micro",
  "accountId" : "123456789012",
  "pendingTime" : "2015-11-19T16:32:11Z",
  "imageId" : "ami-5fb8c835",
  "kernelId" : "aki-919dcaf8",
  "ramdiskId" : null,
  "architecture" : "x86_64"
}`

const validIamInfo = `{
  "Code" : "Success",
  "LastUpdated" : "2016-03-17T12:27:32Z",
  "InstanceProfileArn" : "arn:aws:iam::123456789012:instance-profile/my-instance-profile",
  "InstanceProfileId" : "AIPAABCDEFGHIJKLMN123"
}`

const unsuccessfulIamInfo = `{
  "Code" : "Failed",
  "LastUpdated" : "2016-03-17T12:27:32Z",
  "InstanceProfileArn" : "arn:aws:iam::123456789012:instance-profile/my-instance-profile",
  "InstanceProfileId" : "AIPAABCDEFGHIJKLMN123"
}`

func initTestServer(path string, resp string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.RequestURI != path {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		w.Write([]byte(resp))
	}))
}

func TestEndpoint(t *testing.T) {
	c := ec2metadata.New(unit.Session)
	op := &request.Operation{
		Name:       "GetMetadata",
		HTTPMethod: "GET",
		HTTPPath:   path.Join("/", "meta-data", "testpath"),
	}

	req := c.NewRequest(op, nil, nil)
	if e, a := "http://169.254.169.254/latest", req.ClientInfo.Endpoint; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "http://169.254.169.254/latest/meta-data/testpath", req.HTTPRequest.URL.String(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestGetMetadata(t *testing.T) {
	server := initTestServer(
		"/latest/meta-data/some/path",
		"success", // real response includes suffix
	)
	defer server.Close()
	c := ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")})

	resp, err := c.GetMetadata("some/path")

	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := "success", resp; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestGetUserData(t *testing.T) {
	server := initTestServer(
		"/latest/user-data",
		"success", // real response includes suffix
	)
	defer server.Close()
	c := ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")})

	resp, err := c.GetUserData()

	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := "success", resp; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestGetUserData_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reader := strings.NewReader(`<?xml version="1.0" encoding="iso-8859-1"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
         "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
 <head>
  <title>404 - Not Found</title>
 </head>
 <body>
  <h1>404 - Not Found</h1>
 </body>
</html>`)
		w.Header().Set("Content-Type", "text/html")
		w.Header().Set("Content-Length", fmt.Sprintf("%d", reader.Len()))
		w.WriteHeader(http.StatusNotFound)
		io.Copy(w, reader)
	}))

	defer server.Close()
	c := ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")})

	resp, err := c.GetUserData()
	if err == nil {
		t.Errorf("expect error")
	}
	if len(resp) != 0 {
		t.Errorf("expect empty, got %v", resp)
	}

	aerr := err.(awserr.Error)
	if e, a := "NotFoundError", aerr.Code(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestGetRegion(t *testing.T) {
	server := initTestServer(
		"/latest/meta-data/placement/availability-zone",
		"us-west-2a", // real response includes suffix
	)
	defer server.Close()
	c := ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")})

	region, err := c.Region()

	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := "us-west-2", region; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestMetadataAvailable(t *testing.T) {
	server := initTestServer(
		"/latest/meta-data/instance-id",
		"instance-id",
	)
	defer server.Close()
	c := ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")})

	if !c.Available() {
		t.Errorf("expect available")
	}
}

func TestMetadataIAMInfo_success(t *testing.T) {
	server := initTestServer(
		"/latest/meta-data/iam/info",
		validIamInfo,
	)
	defer server.Close()
	c := ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")})

	iamInfo, err := c.IAMInfo()
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := "Success", iamInfo.Code; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "arn:aws:iam::123456789012:instance-profile/my-instance-profile", iamInfo.InstanceProfileArn; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "AIPAABCDEFGHIJKLMN123", iamInfo.InstanceProfileID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestMetadataIAMInfo_failure(t *testing.T) {
	server := initTestServer(
		"/latest/meta-data/iam/info",
		unsuccessfulIamInfo,
	)
	defer server.Close()
	c := ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")})

	iamInfo, err := c.IAMInfo()
	if err == nil {
		t.Errorf("expect error")
	}
	if e, a := "", iamInfo.Code; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "", iamInfo.InstanceProfileArn; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "", iamInfo.InstanceProfileID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestMetadataNotAvailable(t *testing.T) {
	c := ec2metadata.New(unit.Session)
	c.Handlers.Send.Clear()
	c.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &http.Response{
			StatusCode: int(0),
			Status:     http.StatusText(int(0)),
			Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
		}
		r.Error = awserr.New("RequestError", "send request failed", nil)
		r.Retryable = aws.Bool(true) // network errors are retryable
	})

	if c.Available() {
		t.Errorf("expect not available")
	}
}

func TestMetadataErrorResponse(t *testing.T) {
	c := ec2metadata.New(unit.Session)
	c.Handlers.Send.Clear()
	c.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &http.Response{
			StatusCode: http.StatusBadRequest,
			Status:     http.StatusText(http.StatusBadRequest),
			Body:       ioutil.NopCloser(strings.NewReader("error message text")),
		}
		r.Retryable = aws.Bool(false) // network errors are retryable
	})

	data, err := c.GetMetadata("uri/path")
	if len(data) != 0 {
		t.Errorf("expect empty, got %v", data)
	}
	if e, a := "error message text", err.Error(); !strings.Contains(a, e) {
		t.Errorf("expect %v to be in %v", e, a)
	}
}

func TestEC2RoleProviderInstanceIdentity(t *testing.T) {
	server := initTestServer(
		"/latest/dynamic/instance-identity/document",
		instanceIdentityDocument,
	)
	defer server.Close()
	c := ec2metadata.New(unit.Session, &aws.Config{Endpoint: aws.String(server.URL + "/latest")})

	doc, err := c.GetInstanceIdentityDocument()
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := doc.AccountID, "123456789012"; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := doc.AvailabilityZone, "us-east-1d"; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := doc.Region, "us-east-1"; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}
