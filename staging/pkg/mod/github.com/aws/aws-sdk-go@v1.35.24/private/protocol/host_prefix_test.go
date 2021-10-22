// +build go1.7

package protocol

import (
	"net/http"
	"net/url"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
)

func TestHostPrefixBuilder(t *testing.T) {
	cases := map[string]struct {
		URLHost  string
		ReqHost  string
		Prefix   string
		LabelsFn func() map[string]string
		Disabled bool

		ExpectURLHost string
		ExpectReqHost string
	}{
		"no labels": {
			URLHost:       "service.region.amazonaws.com",
			Prefix:        "data-",
			ExpectURLHost: "data-service.region.amazonaws.com",
		},
		"with labels": {
			URLHost: "service.region.amazonaws.com",
			Prefix:  "{first}-{second}.",
			LabelsFn: func() map[string]string {
				return map[string]string{
					"first":  "abc",
					"second": "123",
				}
			},
			ExpectURLHost: "abc-123.service.region.amazonaws.com",
		},
		"with host prefix disabled": {
			Disabled: true,
			URLHost:  "service.region.amazonaws.com",
			Prefix:   "{first}-{second}.",
			LabelsFn: func() map[string]string {
				return map[string]string{
					"first":  "abc",
					"second": "123",
				}
			},
			ExpectURLHost: "service.region.amazonaws.com",
		},
		"with duplicate labels": {
			URLHost: "service.region.amazonaws.com",
			Prefix:  "{first}-{second}-{first}.",
			LabelsFn: func() map[string]string {
				return map[string]string{
					"first":  "abc",
					"second": "123",
				}
			},
			ExpectURLHost: "abc-123-abc.service.region.amazonaws.com",
		},
		"with unbracketed labels": {
			URLHost: "service.region.amazonaws.com",
			Prefix:  "first-{second}.",
			LabelsFn: func() map[string]string {
				return map[string]string{
					"first":  "abc",
					"second": "123",
				}
			},
			ExpectURLHost: "first-123.service.region.amazonaws.com",
		},
		"with req host": {
			URLHost:       "service.region.amazonaws.com:1234",
			ReqHost:       "service.region.amazonaws.com",
			Prefix:        "data-",
			ExpectURLHost: "data-service.region.amazonaws.com:1234",
			ExpectReqHost: "data-service.region.amazonaws.com",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			builder := HostPrefixBuilder{
				Prefix: c.Prefix, LabelsFn: c.LabelsFn,
			}
			req := &request.Request{
				Config: aws.Config{
					DisableEndpointHostPrefix: aws.Bool(c.Disabled),
				},
				HTTPRequest: &http.Request{
					Host: c.ReqHost,
					URL: &url.URL{
						Host: c.URLHost,
					},
				},
			}

			builder.Build(req)
			if e, a := c.ExpectURLHost, req.HTTPRequest.URL.Host; e != a {
				t.Errorf("expect URL host %v, got %v", e, a)
			}
			if e, a := c.ExpectReqHost, req.HTTPRequest.Host; e != a {
				t.Errorf("expect request host %v, got %v", e, a)
			}
		})
	}
}
