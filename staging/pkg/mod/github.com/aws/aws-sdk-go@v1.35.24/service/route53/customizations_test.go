package route53_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/route53"
)

func TestBuildCorrectURI(t *testing.T) {
	const expectPath = "/2013-04-01/hostedzone/ABCDEFG"

	svc := route53.New(unit.Session)
	svc.Handlers.Validate.Clear()
	req, _ := svc.GetHostedZoneRequest(&route53.GetHostedZoneInput{
		Id: aws.String("/hostedzone/ABCDEFG"),
	})

	req.HTTPRequest.URL.RawQuery = "abc=123"

	req.Build()

	if a, e := req.HTTPRequest.URL.Path, expectPath; a != e {
		t.Errorf("expect path %q, got %q", e, a)
	}

	if a, e := req.HTTPRequest.URL.RawPath, expectPath; a != e {
		t.Errorf("expect raw path %q, got %q", e, a)
	}

	if a, e := req.HTTPRequest.URL.RawQuery, "abc=123"; a != e {
		t.Errorf("expect query to be %q, got %q", e, a)
	}
}
