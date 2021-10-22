// +build go1.7

package ec2metadata_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"path"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/internal/sdktesting"
)

const instanceIdentityDocument = `{
  "devpayProductCodes" : null,
  "marketplaceProductCodes" : [ "1abc2defghijklm3nopqrs4tu" ], 
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

const (
	ttlHeader   = "x-aws-ec2-metadata-token-ttl-seconds"
	tokenHeader = "x-aws-ec2-metadata-token"
)

type testType int

const (
	SecureTestType testType = iota
	InsecureTestType
	BadRequestTestType
	ServerErrorForTokenTestType
	pageNotFoundForTokenTestType
	pageNotFoundWith401TestType
)

type testServer struct {
	t *testing.T

	tokens      []string
	activeToken atomic.Value
	data        string
}

type operationListProvider struct {
	operationsPerformed []string
}

func getTokenRequiredParams(t *testing.T, fn http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if e, a := "PUT", r.Method; e != a {
			t.Errorf("expect %v, http method got %v", e, a)
			http.Error(w, "wrong method", 400)
			return
		}
		if len(r.Header.Get(ttlHeader)) == 0 {
			t.Errorf("expect ttl header to be present in the request headers, got none")
			http.Error(w, "wrong method", 400)
			return
		}

		fn(w, r)
	}
}

func newTestServer(t *testing.T, testType testType, testServer *testServer) *httptest.Server {
	mux := http.NewServeMux()
	switch testType {
	case SecureTestType:
		mux.HandleFunc("/latest/api/token", getTokenRequiredParams(t, testServer.secureGetTokenHandler))
		mux.HandleFunc("/", testServer.secureGetLatestHandler)
	case InsecureTestType:
		mux.HandleFunc("/latest/api/token", testServer.insecureGetTokenHandler)
		mux.HandleFunc("/", testServer.insecureGetLatestHandler)
	case BadRequestTestType:
		mux.HandleFunc("/latest/api/token", getTokenRequiredParams(t, testServer.badRequestGetTokenHandler))
		mux.HandleFunc("/", testServer.badRequestGetLatestHandler)
	case ServerErrorForTokenTestType:
		mux.HandleFunc("/latest/api/token", getTokenRequiredParams(t, testServer.serverErrorGetTokenHandler))
		mux.HandleFunc("/", testServer.insecureGetLatestHandler)
	case pageNotFoundForTokenTestType:
		mux.HandleFunc("/latest/api/token", getTokenRequiredParams(t, testServer.pageNotFoundGetTokenHandler))
		mux.HandleFunc("/", testServer.insecureGetLatestHandler)
	case pageNotFoundWith401TestType:
		mux.HandleFunc("/latest/api/token", getTokenRequiredParams(t, testServer.pageNotFoundGetTokenHandler))
		mux.HandleFunc("/", testServer.unauthorizedGetLatestHandler)

	}

	return httptest.NewServer(mux)
}

func (s *testServer) secureGetTokenHandler(w http.ResponseWriter, r *http.Request) {
	token := s.tokens[0]

	// set the active token
	s.activeToken.Store(token)

	// rotate the token
	if len(s.tokens) > 1 {
		s.tokens = s.tokens[1:]
	}

	// set the header and response body
	w.Header().Set(ttlHeader, r.Header.Get(ttlHeader))
	if activeToken, ok := s.activeToken.Load().(string); ok {
		w.Write([]byte(activeToken))
	} else {
		s.t.Fatalf("Expected activeToken to be of type string, got %v", activeToken)
	}
}

func (s *testServer) secureGetLatestHandler(w http.ResponseWriter, r *http.Request) {
	if s.activeToken.Load() == nil {
		s.t.Errorf("expect token to have been requested, was not")
		http.Error(w, "", 401)
		return
	}

	if e, a := s.activeToken.Load(), r.Header.Get(tokenHeader); e != a {
		s.t.Errorf("expect %v token, got %v", e, a)
		http.Error(w, "", 401)
		return
	}

	w.Header().Set(ttlHeader, r.Header.Get(ttlHeader))
	w.Write([]byte(s.data))
}

func (s *testServer) insecureGetTokenHandler(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "", 404)
}

func (s *testServer) insecureGetLatestHandler(w http.ResponseWriter, r *http.Request) {
	if len(r.Header.Get(tokenHeader)) != 0 {
		s.t.Errorf("Request token found, expected none")
		http.Error(w, "", 400)
		return
	}

	w.Write([]byte(s.data))
}

func (s *testServer) badRequestGetTokenHandler(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "", 400)
}

func (s *testServer) badRequestGetLatestHandler(w http.ResponseWriter, r *http.Request) {
	s.t.Errorf("Expected no call to this handler, incorrect behavior found")
}

func (s *testServer) serverErrorGetTokenHandler(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "", 403)
}

func (s *testServer) pageNotFoundGetTokenHandler(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "Page not found error", 404)
}

func (s *testServer) unauthorizedGetLatestHandler(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "", 401)
}

func (opListProvider *operationListProvider) addToOperationPerformedList(r *request.Request) {
	opListProvider.operationsPerformed = append(opListProvider.operationsPerformed, r.Operation.Name)
}

func TestEndpoint(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	c := ec2metadata.New(unit.Session)
	op := &request.Operation{
		Name:       "GetMetadata",
		HTTPMethod: "GET",
		HTTPPath:   path.Join("/latest", "meta-data", "testpath"),
	}

	req := c.NewRequest(op, nil, nil)
	if e, a := "http://169.254.169.254/latest/meta-data/testpath", req.HTTPRequest.URL.String(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestGetMetadata(t *testing.T) {
	cases := map[string]struct {
		NewServer                   func(t *testing.T) *httptest.Server
		expectedData                string
		expectedError               string
		expectedOperationsPerformed []string
	}{
		"Insecure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := InsecureTestType
				Ts := &testServer{
					t:    t,
					data: "IMDSProfileForGoSDK",
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                "IMDSProfileForGoSDK",
			expectedOperationsPerformed: []string{"GetToken", "GetMetadata", "GetMetadata"},
		},
		"Secure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := SecureTestType
				Ts := &testServer{
					t:      t,
					tokens: []string{"firstToken", "secondToken", "thirdToken"},
					data:   "IMDSProfileForGoSDK",
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                "IMDSProfileForGoSDK",
			expectedError:               "",
			expectedOperationsPerformed: []string{"GetToken", "GetMetadata", "GetMetadata"},
		},
		"Bad request case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := BadRequestTestType
				Ts := &testServer{
					t:      t,
					tokens: []string{"firstToken", "secondToken", "thirdToken"},
					data:   "IMDSProfileForGoSDK",
				}
				return newTestServer(t, testType, Ts)
			},
			expectedError:               "400",
			expectedOperationsPerformed: []string{"GetToken", "GetMetadata", "GetToken", "GetMetadata"},
		},
		"ServerErrorForTokenTestType": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := ServerErrorForTokenTestType
				Ts := &testServer{
					t:      t,
					tokens: []string{},
					data:   "IMDSProfileForGoSDK",
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                "IMDSProfileForGoSDK",
			expectedOperationsPerformed: []string{"GetToken", "GetMetadata", "GetMetadata"},
		},
	}

	for name, x := range cases {
		t.Run(name, func(t *testing.T) {

			server := x.NewServer(t)
			defer server.Close()

			op := &operationListProvider{}

			c := ec2metadata.New(unit.Session, &aws.Config{
				Endpoint: aws.String(server.URL),
			})
			c.Handlers.Complete.PushBack(op.addToOperationPerformedList)

			resp, err := c.GetMetadata("some/path")

			// token should stay alive, since default duration is 26000 seconds
			resp, err = c.GetMetadata("some/path")

			if len(x.expectedError) != 0 {
				if err == nil {
					t.Fatalf("expect %v error, got none", x.expectedError)
				}
				if e, a := x.expectedError, err.Error(); !strings.Contains(a, e) {
					t.Fatalf("expect %v error, got %v", e, a)
				}
			} else if err != nil {
				t.Fatalf("expect no error, got %v", err)
			}

			if e, a := x.expectedData, resp; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}

			if e, a := x.expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
				t.Errorf("expect %v operations, got %v", e, a)
			}

		})
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
	c := ec2metadata.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	})

	resp, err := c.GetUserData()
	if err == nil {
		t.Fatalf("expect error")
	}
	if len(resp) != 0 {
		t.Fatalf("expect empty, got %v", resp)
	}

	if requestFailedError, ok := err.(awserr.RequestFailure); ok {
		if e, a := http.StatusNotFound, requestFailedError.StatusCode(); e != a {
			t.Fatalf("expect %v, got %v", e, a)
		}
	}
}

func TestGetRegion(t *testing.T) {
	cases := map[string]struct {
		NewServer                   func(t *testing.T) *httptest.Server
		expectedData                string
		expectedError               string
		expectedOperationsPerformed []string
	}{
		"Insecure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := InsecureTestType
				Ts := &testServer{
					t:    t,
					data: instanceIdentityDocument,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                "us-east-1",
			expectedOperationsPerformed: []string{"GetToken", "GetDynamicData"},
		},
		"Secure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := SecureTestType
				Ts := &testServer{
					t:      t,
					tokens: []string{"firstToken", "secondToken", "thirdToken"},
					data:   instanceIdentityDocument,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                "us-east-1",
			expectedOperationsPerformed: []string{"GetToken", "GetDynamicData"},
		},
		"Bad request case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := BadRequestTestType
				Ts := &testServer{
					t:      t,
					tokens: []string{"firstToken", "secondToken", "thirdToken"},
					data:   instanceIdentityDocument,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedError:               "400",
			expectedOperationsPerformed: []string{"GetToken", "GetDynamicData"},
		},
		"ServerErrorForTokenTestType": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := ServerErrorForTokenTestType
				Ts := &testServer{
					t:      t,
					tokens: []string{},
					data:   instanceIdentityDocument,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                "us-east-1",
			expectedOperationsPerformed: []string{"GetToken", "GetDynamicData"},
		},
	}

	for name, x := range cases {
		t.Run(name, func(t *testing.T) {

			server := x.NewServer(t)
			defer server.Close()

			op := &operationListProvider{}

			c := ec2metadata.New(unit.Session, &aws.Config{
				Endpoint: aws.String(server.URL),
			})
			c.Handlers.Complete.PushBack(op.addToOperationPerformedList)

			resp, err := c.Region()

			if len(x.expectedError) != 0 {
				if err == nil {
					t.Fatalf("expect %v error, got none", x.expectedError)
				}
				if e, a := x.expectedError, err.Error(); !strings.Contains(a, e) {
					t.Fatalf("expect %v error, got %v", e, a)
				}
			} else if err != nil {
				t.Fatalf("expect no error, got %v", err)
			}

			if e, a := x.expectedData, resp; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}

			if e, a := x.expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
				t.Fatalf("expect %v operations, got %v", e, a)
			}
		})
	}
}

func TestMetadataIAMInfo_success(t *testing.T) {
	cases := map[string]struct {
		NewServer                   func(t *testing.T) *httptest.Server
		expectedData                string
		expectedError               string
		expectedOperationsPerformed []string
	}{
		"Insecure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := InsecureTestType
				Ts := &testServer{
					t:    t,
					data: validIamInfo,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                validIamInfo,
			expectedOperationsPerformed: []string{"GetToken", "GetMetadata"},
		},
		"Secure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := SecureTestType
				Ts := &testServer{
					t:      t,
					tokens: []string{"firstToken", "secondToken", "thirdToken"},
					data:   validIamInfo,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                validIamInfo,
			expectedOperationsPerformed: []string{"GetToken", "GetMetadata"},
		},
	}

	for name, x := range cases {
		t.Run(name, func(t *testing.T) {

			server := x.NewServer(t)
			defer server.Close()

			op := &operationListProvider{}

			c := ec2metadata.New(unit.Session, &aws.Config{
				Endpoint: aws.String(server.URL),
			})
			c.Handlers.Complete.PushBack(op.addToOperationPerformedList)

			iamInfo, err := c.IAMInfo()

			if len(x.expectedError) != 0 {
				if err == nil {
					t.Fatalf("expect %v error, got none", x.expectedError)
				}
				if e, a := x.expectedError, err.Error(); !strings.Contains(a, e) {
					t.Fatalf("expect %v error, got %v", e, a)
				}
			} else if err != nil {
				t.Fatalf("expect no error, got %v", err)
			}

			if e, a := "Success", iamInfo.Code; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}
			if e, a := "arn:aws:iam::123456789012:instance-profile/my-instance-profile", iamInfo.InstanceProfileArn; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}
			if e, a := "AIPAABCDEFGHIJKLMN123", iamInfo.InstanceProfileID; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}

			if e, a := x.expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
				t.Fatalf("expect %v operations, got %v", e, a)
			}
		})
	}
}

func TestMetadataIAMInfo_failure(t *testing.T) {
	cases := map[string]struct {
		NewServer                   func(t *testing.T) *httptest.Server
		expectedData                string
		expectedError               string
		expectedOperationsPerformed []string
	}{
		"Insecure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := InsecureTestType
				Ts := &testServer{
					t:      t,
					tokens: nil,
					data:   unsuccessfulIamInfo,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                unsuccessfulIamInfo,
			expectedOperationsPerformed: []string{"GetToken", "GetMetadata"},
		},
		"Secure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := SecureTestType
				Ts := &testServer{
					t:      t,
					tokens: []string{"firstToken", "secondToken", "thirdToken"},
					data:   unsuccessfulIamInfo,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                unsuccessfulIamInfo,
			expectedOperationsPerformed: []string{"GetToken", "GetMetadata"},
		},
	}

	for name, x := range cases {
		t.Run(name, func(t *testing.T) {

			server := x.NewServer(t)
			defer server.Close()

			op := &operationListProvider{}

			c := ec2metadata.New(unit.Session, &aws.Config{
				Endpoint: aws.String(server.URL),
			})
			c.Handlers.Complete.PushBack(op.addToOperationPerformedList)

			iamInfo, err := c.IAMInfo()
			if err == nil {
				t.Fatalf("expect error")
			}
			if e, a := "", iamInfo.Code; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}
			if e, a := "", iamInfo.InstanceProfileArn; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}
			if e, a := "", iamInfo.InstanceProfileID; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}
			if e, a := x.expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
				t.Fatalf("expect %v operations, got %v", e, a)
			}
		})
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
		r.Error = awserr.New(request.ErrCodeRequestError, "send request failed", nil)
		r.Retryable = aws.Bool(true) // network errors are retryable
	})

	if c.Available() {
		t.Fatalf("expect not available")
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
	if e, a := "error message text", err.Error(); !strings.Contains(a, e) {
		t.Fatalf("expect %v to be in %v", e, a)
	}
	if len(data) != 0 {
		t.Fatalf("expect empty, got %v", data)
	}

}

func TestEC2RoleProviderInstanceIdentity(t *testing.T) {
	cases := map[string]struct {
		NewServer                   func(t *testing.T) *httptest.Server
		expectedData                string
		expectedOperationsPerformed []string
	}{
		"Insecure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := InsecureTestType
				Ts := &testServer{
					t:      t,
					tokens: nil,
					data:   instanceIdentityDocument,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                instanceIdentityDocument,
			expectedOperationsPerformed: []string{"GetToken", "GetDynamicData"},
		},
		"Secure server success case": {
			NewServer: func(t *testing.T) *httptest.Server {
				testType := SecureTestType
				Ts := &testServer{
					t:      t,
					tokens: []string{"firstToken", "secondToken", "thirdToken"},
					data:   instanceIdentityDocument,
				}
				return newTestServer(t, testType, Ts)
			},
			expectedData:                instanceIdentityDocument,
			expectedOperationsPerformed: []string{"GetToken", "GetDynamicData"},
		},
	}

	for name, x := range cases {
		t.Run(name, func(t *testing.T) {

			server := x.NewServer(t)
			defer server.Close()

			op := &operationListProvider{}

			c := ec2metadata.New(unit.Session, &aws.Config{
				Endpoint: aws.String(server.URL),
			})
			c.Handlers.Complete.PushBack(op.addToOperationPerformedList)
			doc, err := c.GetInstanceIdentityDocument()

			if err != nil {
				t.Fatalf("expected no error, got %v", err)
			}

			if e, a := doc.AccountID, "123456789012"; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}
			if e, a := doc.AvailabilityZone, "us-east-1d"; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}
			if e, a := doc.Region, "us-east-1"; e != a {
				t.Fatalf("expect %v, got %v", e, a)
			}
			if e, a := x.expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
				t.Fatalf("expect %v operations, got %v", e, a)
			}
		})
	}
}

func TestEC2MetadataRetryFailure(t *testing.T) {
	mux := http.NewServeMux()

	mux.HandleFunc("/latest/api/token", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "PUT" && r.Header.Get(ttlHeader) != "" {
			w.Header().Set(ttlHeader, "200")
			http.Error(w, "service unavailable", http.StatusServiceUnavailable)
			return
		}
		http.Error(w, "bad request", http.StatusBadRequest)
	})

	// meta-data endpoint for this test, just returns the token
	mux.HandleFunc("/latest/meta-data/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("profile_name"))
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	c := ec2metadata.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	})

	c.Handlers.AfterRetry.PushBack(func(i *request.Request) {
		t.Logf("%v received, retrying operation %v", i.HTTPResponse.StatusCode, i.Operation.Name)
	})
	c.Handlers.Complete.PushBack(func(i *request.Request) {
		t.Logf("%v operation exited with status %v", i.Operation.Name, i.HTTPResponse.StatusCode)
	})

	resp, err := c.GetMetadata("some/path")
	if err != nil {
		t.Fatalf("Expected none, got error %v", err)
	}
	if resp != "profile_name" {
		t.Fatalf("Expected response to be profile_name, got %v", resp)
	}

	resp, err = c.GetMetadata("some/path")
	if err != nil {
		t.Fatalf("Expected none, got error %v", err)
	}
	if resp != "profile_name" {
		t.Fatalf("Expected response to be profile_name, got %v", resp)
	}
}

func TestEC2MetadataRetryOnce(t *testing.T) {
	var secureDataFlow bool
	var retry = true
	mux := http.NewServeMux()

	mux.HandleFunc("/latest/api/token", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "PUT" && r.Header.Get(ttlHeader) != "" {
			w.Header().Set(ttlHeader, "200")
			for retry {
				retry = false
				http.Error(w, "service unavailable", http.StatusServiceUnavailable)
				return
			}
			w.Write([]byte("token"))
			secureDataFlow = true
			return
		}
		http.Error(w, "bad request", http.StatusBadRequest)
	})

	// meta-data endpoint for this test, just returns the token
	mux.HandleFunc("/latest/meta-data/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(r.Header.Get(tokenHeader)))
	})

	var tokenRetryCount int

	server := httptest.NewServer(mux)
	defer server.Close()
	c := ec2metadata.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	})

	// Handler on client that logs if retried
	c.Handlers.AfterRetry.PushBack(func(i *request.Request) {
		t.Logf("%v received, retrying operation %v", i.HTTPResponse.StatusCode, i.Operation.Name)
		tokenRetryCount++
	})

	_, err := c.GetMetadata("some/path")

	if tokenRetryCount != 1 {
		t.Fatalf("Expected number of retries for fetching token to be 1, got %v", tokenRetryCount)
	}

	if !secureDataFlow {
		t.Fatalf("Expected secure data flow to be %v, got %v", secureDataFlow, !secureDataFlow)
	}

	if err != nil {
		t.Fatalf("Expected none, got error %v", err)
	}
}

func TestEC2Metadata_Concurrency(t *testing.T) {
	ts := &testServer{
		t:      t,
		tokens: []string{"firstToken"},
		data:   "IMDSProfileForSDKGo",
	}

	server := newTestServer(t, SecureTestType, ts)
	defer server.Close()

	c := ec2metadata.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	})

	var wg sync.WaitGroup
	wg.Add(10)
	for i := 0; i < 10; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				resp, err := c.GetMetadata("some/data")
				if err != nil {
					t.Errorf("expect no error, got %v", err)
				}

				if e, a := "IMDSProfileForSDKGo", resp; e != a {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		}()
	}
	wg.Wait()
}

func TestRequestOnMetadata(t *testing.T) {
	ts := &testServer{
		t:      t,
		tokens: []string{"firstToken", "secondToken"},
		data:   "profile_name",
	}
	server := newTestServer(t, SecureTestType, ts)
	defer server.Close()

	c := ec2metadata.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	})
	req := c.NewRequest(&request.Operation{
		Name:            "Ec2Metadata request",
		HTTPMethod:      "GET",
		HTTPPath:        "/latest/foo",
		Paginator:       nil,
		BeforePresignFn: nil,
	}, nil, nil)

	op := &operationListProvider{}
	c.Handlers.Complete.PushBack(op.addToOperationPerformedList)
	err := req.Send()

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	if len(op.operationsPerformed) < 1 {
		t.Fatalf("Expected atleast one operation GetToken to be called on EC2Metadata client")
		return
	}

	if op.operationsPerformed[0] != "GetToken" {
		t.Fatalf("Expected GetToken operation to be called")
	}

}

func TestExhaustiveRetryToFetchToken(t *testing.T) {
	ts := &testServer{
		t:      t,
		tokens: []string{"firstToken", "secondToken"},
		data:   "IMDSProfileForSDKGo",
	}

	server := newTestServer(t, pageNotFoundForTokenTestType, ts)
	defer server.Close()

	op := &operationListProvider{}

	c := ec2metadata.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	})
	c.Handlers.Complete.PushBack(op.addToOperationPerformedList)

	resp, err := c.GetMetadata("/some/path")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if e, a := "IMDSProfileForSDKGo", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}

	resp, err = c.GetMetadata("/some/path")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if e, a := "IMDSProfileForSDKGo", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}

	resp, err = c.GetMetadata("/some/path")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if e, a := "IMDSProfileForSDKGo", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}

	resp, err = c.GetMetadata("/some/path")
	expectedOperationsPerformed := []string{"GetToken", "GetMetadata", "GetMetadata", "GetMetadata", "GetMetadata"}
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if e, a := "IMDSProfileForSDKGo", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	if e, a := expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
		t.Fatalf("expect %v operations, got %v", e, a)
	}
}

func TestExhaustiveRetryWith401(t *testing.T) {
	ts := &testServer{
		t:      t,
		tokens: []string{"firstToken", "secondToken"},
		data:   "IMDSProfileForSDKGo",
	}

	server := newTestServer(t, pageNotFoundWith401TestType, ts)
	defer server.Close()

	op := &operationListProvider{}

	c := ec2metadata.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	})
	c.Handlers.Complete.PushBack(op.addToOperationPerformedList)

	resp, err := c.GetMetadata("/some/path")
	if err == nil {
		t.Fatalf("Expected %v error, got none", err)
	}
	if e, a := "", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	resp, err = c.GetMetadata("/some/path")
	if err == nil {
		t.Fatalf("Expected %v error, got none", err)
	}
	if e, a := "", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	resp, err = c.GetMetadata("/some/path")
	if err == nil {
		t.Fatalf("Expected %v error, got none", err)
	}
	if e, a := "", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	resp, err = c.GetMetadata("/some/path")

	expectedOperationsPerformed := []string{"GetToken", "GetMetadata", "GetToken", "GetMetadata", "GetToken", "GetMetadata", "GetToken", "GetMetadata"}

	if err == nil {
		t.Fatalf("Expected %v error, got none", err)
	}
	if e, a := "", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	if e, a := expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
		t.Fatalf("expect %v operations, got %v", e, a)
	}
}

func TestRequestTimeOut(t *testing.T) {
	mux := http.NewServeMux()
	done := make(chan bool)
	mux.HandleFunc("/latest/api/token", func(w http.ResponseWriter, r *http.Request) {
		// wait to read from channel done
		<-done
	})

	mux.HandleFunc("/latest/", func(w http.ResponseWriter, r *http.Request) {
		if len(r.Header.Get(tokenHeader)) != 0 {
			http.Error(w, "", 400)
			return
		}
		w.Write([]byte("IMDSProfileForSDKGo"))
	})

	server := httptest.NewServer(mux)
	defer server.Close()
	defer close(done)

	op := &operationListProvider{}

	c := ec2metadata.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	})
	// for test, change the timeout to 100 ms
	c.Config.HTTPClient.Timeout = 100 * time.Millisecond

	c.Handlers.Complete.PushBack(op.addToOperationPerformedList)

	start := time.Now()
	resp, err := c.GetMetadata("/some/path")

	if e, a := 1*time.Second, time.Since(start); e < a {
		t.Fatalf("expected duration of test to be less than %v, got %v", e, a)
	}

	if e, a := "IMDSProfileForSDKGo", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	expectedOperationsPerformed := []string{"GetToken", "GetMetadata"}
	if e, a := expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
		t.Fatalf("expect %v operations, got %v", e, a)
	}

	start = time.Now()
	resp, err = c.GetMetadata("/some/path")
	if e, a := 1*time.Second, time.Since(start); e < a {
		t.Fatalf("expected duration of test to be less than %v, got %v", e, a)
	}

	if e, a := "IMDSProfileForSDKGo", resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	expectedOperationsPerformed = []string{"GetToken", "GetMetadata", "GetMetadata"}
	if e, a := expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
		t.Fatalf("expect %v operations, got %v", e, a)
	}
}

func TestTokenExpiredBehavior(t *testing.T) {
	tokens := []string{"firstToken", "secondToken", "thirdToken"}
	var activeToken string
	mux := http.NewServeMux()

	mux.HandleFunc("/latest/api/token", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "PUT" && r.Header.Get(ttlHeader) != "" {
			// set ttl to 0, so TTL is expired.
			w.Header().Set(ttlHeader, "0")
			activeToken = tokens[0]
			if len(tokens) > 1 {
				tokens = tokens[1:]
			}

			w.Write([]byte(activeToken))
			return
		}
		http.Error(w, "bad request", http.StatusBadRequest)
	})

	// meta-data endpoint for this test, just returns the token
	mux.HandleFunc("/latest/meta-data/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(ttlHeader, r.Header.Get(ttlHeader))
		w.Write([]byte(r.Header.Get(tokenHeader)))
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	op := &operationListProvider{}

	c := ec2metadata.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	})
	c.Handlers.Complete.PushBack(op.addToOperationPerformedList)

	resp, err := c.GetMetadata("/some/path")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if e, a := activeToken, resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}

	// store the token received before
	var firstToken = activeToken

	resp, err = c.GetMetadata("/some/path")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if e, a := activeToken, resp; e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}

	// Since TTL is 0, we should have received a new token
	if firstToken == activeToken {
		t.Fatalf("Expected token should have expired, and not the same")
	}

	expectedOperationsPerformed := []string{"GetToken", "GetMetadata", "GetToken", "GetMetadata"}

	if e, a := expectedOperationsPerformed, op.operationsPerformed; !reflect.DeepEqual(e, a) {
		t.Fatalf("expect %v operations, got %v", e, a)
	}
}
