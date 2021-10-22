// +build go1.7

package awsendpointdiscoverytest

import (
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

func TestEndpointDiscoveryWithCustomEndpoint(t *testing.T) {
	mockEndpointResolver := endpoints.ResolverFunc(func(service string, region string, opts ...func(options *endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
		return endpoints.ResolvedEndpoint{
			URL: "https://mockEndpointForDiscovery",
		}, nil
	})

	cases := map[string]struct {
		hasDiscoveryEnabled bool
		hasCustomEndpoint   bool
		isOperationRequired bool
		customEndpoint      string
		expectedEndpoint    string
	}{
		"HasCustomEndpoint_RequiredOperation": {
			hasDiscoveryEnabled: true,
			hasCustomEndpoint:   true,
			isOperationRequired: true,
			customEndpoint:      "https://mockCustomEndpoint",
			expectedEndpoint:    "https://mockCustomEndpoint/",
		},
		"HasCustomEndpoint_OptionalOperation": {
			hasDiscoveryEnabled: true,
			hasCustomEndpoint:   true,
			customEndpoint:      "https://mockCustomEndpoint",
			expectedEndpoint:    "https://mockCustomEndpoint/",
		},
		"NoCustomEndpoint_DiscoveryDisabled": {
			expectedEndpoint: "https://mockEndpointForDiscovery/",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := &aws.Config{
				EnableEndpointDiscovery: aws.Bool(c.hasDiscoveryEnabled),
				EndpointResolver:        mockEndpointResolver,
			}
			if c.hasCustomEndpoint {
				cfg.Endpoint = aws.String(c.customEndpoint)
			}

			svc := New(unit.Session, cfg)
			svc.Handlers.Clear()
			// Add a handler to verify no call goes to DescribeEndpoints operation
			svc.Handlers.Send.PushBack(func(r *request.Request) {
				if ne, a := opDescribeEndpoints, r.Operation.Name; strings.EqualFold(ne, a) {
					t.Errorf("expected no call to %q operation", a)
				}
			})

			var req *request.Request
			if c.isOperationRequired {
				req, _ = svc.TestDiscoveryIdentifiersRequiredRequest(
					&TestDiscoveryIdentifiersRequiredInput{
						Sdk: aws.String("sdk"),
					},
				)
			} else {
				req, _ = svc.TestDiscoveryOptionalRequest(
					&TestDiscoveryOptionalInput{
						Sdk: aws.String("sdk"),
					},
				)
			}

			req.Handlers.Send.PushBack(func(r *request.Request) {
				if e, a := c.expectedEndpoint, r.HTTPRequest.URL.String(); e != a {
					t.Errorf("expected %q, but received %q", e, a)
				}
			})
			if err := req.Send(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEndpointDiscoveryWithAttemptedDiscovery(t *testing.T) {
	mockEndpointResolver := endpoints.ResolverFunc(func(service string, region string, opts ...func(options *endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
		return endpoints.ResolvedEndpoint{
			URL: "https://mockEndpointForDiscovery",
		}, nil
	})

	cases := map[string]struct {
		hasDiscoveryEnabled bool
		hasCustomEndpoint   bool
		isOperationRequired bool
		customEndpoint      string
		expectedEndpoint    string
	}{
		"NoCustomEndpoint_RequiredOperation": {
			hasDiscoveryEnabled: true,
			isOperationRequired: true,
			expectedEndpoint:    "https://mockEndpointForDiscovery/",
		},
		"NoCustomEndpoint_OptionalOperation": {
			hasDiscoveryEnabled: true,
			expectedEndpoint:    "https://mockEndpointForDiscovery/",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := &aws.Config{
				EnableEndpointDiscovery: aws.Bool(c.hasDiscoveryEnabled),
				EndpointResolver:        mockEndpointResolver,
			}
			if c.hasCustomEndpoint {
				cfg.Endpoint = aws.String(c.customEndpoint)
			}

			svc := New(unit.Session, cfg)
			svc.Handlers.Clear()
			req, _ := svc.TestDiscoveryIdentifiersRequiredRequest(
				&TestDiscoveryIdentifiersRequiredInput{
					Sdk: aws.String("sdk"),
				},
			)

			svc.Handlers.Send.PushBack(func(r *request.Request) {
				if e, a := opDescribeEndpoints, r.Operation.Name; e != a {
					t.Fatalf("expected operaton to be %q, called %q instead", e, a)
				}
			})

			req.Handlers.Send.PushBack(func(r *request.Request) {
				if e, a := c.expectedEndpoint, r.HTTPRequest.URL.String(); e != a {
					t.Errorf("expected %q, but received %q", e, a)
				}
			})

			if err := req.Send(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEndpointDiscovery(t *testing.T) {
	svc := New(unit.Session, &aws.Config{
		EnableEndpointDiscovery: aws.Bool(true),
	})
	svc.Handlers.Clear()
	svc.Handlers.Send.PushBack(mockSendDescEndpoint("http://foo"))

	var descCount int32
	svc.Handlers.Complete.PushBack(func(r *request.Request) {
		if r.Operation.Name != opDescribeEndpoints {
			return
		}
		atomic.AddInt32(&descCount, 1)
	})

	for i := 0; i < 2; i++ {
		req, _ := svc.TestDiscoveryIdentifiersRequiredRequest(
			&TestDiscoveryIdentifiersRequiredInput{
				Sdk: aws.String("sdk"),
			},
		)
		req.Handlers.Send.PushBack(func(r *request.Request) {
			if e, a := "http://foo", r.HTTPRequest.URL.String(); e != a {
				t.Errorf("expected %q, but received %q", e, a)
			}
		})
		if err := req.Send(); err != nil {
			t.Fatal(err)
		}
	}

	if e, a := int32(1), atomic.LoadInt32(&descCount); e != a {
		t.Errorf("expect desc endpoint called %d, got %d", e, a)
	}
}

func TestAsyncEndpointDiscovery(t *testing.T) {
	t.Parallel()

	svc := New(unit.Session, &aws.Config{
		EnableEndpointDiscovery: aws.Bool(true),
	})
	svc.Handlers.Clear()

	var firstAsyncReq sync.WaitGroup
	firstAsyncReq.Add(1)
	svc.Handlers.Build.PushBack(func(r *request.Request) {
		if r.Operation.Name == opDescribeEndpoints {
			firstAsyncReq.Wait()
		}
	})
	svc.Handlers.Send.PushBack(mockSendDescEndpoint("http://foo"))

	req, _ := svc.TestDiscoveryOptionalRequest(&TestDiscoveryOptionalInput{
		Sdk: aws.String("sdk"),
	})
	const clientHost = "awsendpointdiscoverytestservice.mock-region.amazonaws.com"
	req.Handlers.Send.PushBack(func(r *request.Request) {
		if e, a := clientHost, r.HTTPRequest.URL.Host; e != a {
			t.Errorf("expected %q, but received %q", e, a)
		}
	})
	req.Handlers.Complete.PushBack(func(r *request.Request) {
		firstAsyncReq.Done()
	})
	if err := req.Send(); err != nil {
		t.Fatal(err)
	}

	var cacheUpdated bool
	for s := time.Now().Add(10 * time.Second); s.After(time.Now()); {
		// Wait for the cache to be updated before making second request.
		if svc.endpointCache.Has(req.Operation.Name) {
			cacheUpdated = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !cacheUpdated {
		t.Fatalf("expect endpoint cache to be updated, was not")
	}

	req, _ = svc.TestDiscoveryOptionalRequest(&TestDiscoveryOptionalInput{
		Sdk: aws.String("sdk"),
	})
	req.Handlers.Send.PushBack(func(r *request.Request) {
		if e, a := "http://foo", r.HTTPRequest.URL.String(); e != a {
			t.Errorf("expected %q, but received %q", e, a)
		}
	})
	if err := req.Send(); err != nil {
		t.Fatal(err)
	}
}

func TestEndpointDiscovery_EndpointScheme(t *testing.T) {
	cases := []struct {
		address         string
		expectedAddress string
		err             string
	}{
		0: {
			address:         "https://foo",
			expectedAddress: "https://foo",
		},
		1: {
			address:         "bar",
			expectedAddress: "https://bar",
		},
	}

	for i, c := range cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			svc := New(unit.Session, &aws.Config{
				EnableEndpointDiscovery: aws.Bool(true),
			})
			svc.Handlers.Clear()
			svc.Handlers.Send.PushBack(mockSendDescEndpoint(c.address))

			for i := 0; i < 2; i++ {
				req, _ := svc.TestDiscoveryIdentifiersRequiredRequest(
					&TestDiscoveryIdentifiersRequiredInput{
						Sdk: aws.String("sdk"),
					},
				)
				req.Handlers.Send.PushBack(func(r *request.Request) {
					if len(c.err) == 0 {
						if e, a := c.expectedAddress, r.HTTPRequest.URL.String(); e != a {
							t.Errorf("expected %q, but received %q", e, a)
						}
					}
				})

				err := req.Send()
				if err != nil && len(c.err) == 0 {
					t.Fatalf("expected no error, got %v", err)
				} else if err == nil && len(c.err) > 0 {
					t.Fatalf("expected error, got none")
				} else if err != nil && len(c.err) > 0 {
					if e, a := c.err, err.Error(); !strings.Contains(a, e) {
						t.Fatalf("expected %v, got %v", c.err, err)
					}
				}
			}
		})
	}
}

func removeHandlers(h request.Handlers, removeSendHandlers bool) request.Handlers {
	if removeSendHandlers {
		h.Send.Clear()
	}
	h.Unmarshal.Clear()
	h.UnmarshalStream.Clear()
	h.UnmarshalMeta.Clear()
	h.UnmarshalError.Clear()
	h.Validate.Clear()
	h.Complete.Clear()
	h.ValidateResponse.Clear()
	return h
}

func mockSendDescEndpoint(address string) func(r *request.Request) {
	return func(r *request.Request) {
		if r.Operation.Name != opDescribeEndpoints {
			return
		}

		out, _ := r.Data.(*DescribeEndpointsOutput)
		out.Endpoints = []*Endpoint{
			{
				Address:              &address,
				CachePeriodInMinutes: aws.Int64(5),
			},
		}
		r.Data = out
	}
}
