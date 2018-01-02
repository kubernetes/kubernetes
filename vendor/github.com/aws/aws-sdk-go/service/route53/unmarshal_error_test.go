package route53_test

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/route53"
)

func makeClientWithResponse(response string) *route53.Route53 {
	r := route53.New(unit.Session)
	r.Handlers.Send.Clear()
	r.Handlers.Send.PushBack(func(r *request.Request) {
		body := ioutil.NopCloser(bytes.NewReader([]byte(response)))
		r.HTTPResponse = &http.Response{
			ContentLength: int64(len(response)),
			StatusCode:    400,
			Status:        "Bad Request",
			Body:          body,
		}
	})

	return r
}

func TestUnmarshalStandardError(t *testing.T) {
	const errorResponse = `<?xml version="1.0" encoding="UTF-8"?>
<ErrorResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/">
  <Error>
    <Code>InvalidDomainName</Code>
    <Message>The domain name is invalid</Message>
  </Error>
  <RequestId>12345</RequestId>
</ErrorResponse>
`

	r := makeClientWithResponse(errorResponse)

	_, err := r.CreateHostedZone(&route53.CreateHostedZoneInput{
		CallerReference: aws.String("test"),
		Name:            aws.String("test_zone"),
	})

	if err == nil {
		t.Error("expected error, but received none")
	}

	if e, a := "InvalidDomainName", err.(awserr.Error).Code(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}

	if e, a := "The domain name is invalid", err.(awserr.Error).Message(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
}

func TestUnmarshalInvalidChangeBatch(t *testing.T) {
	const errorMessage = `
Tried to create resource record set duplicate.example.com. type A,
but it already exists
`
	const errorResponse = `<?xml version="1.0" encoding="UTF-8"?>
<InvalidChangeBatch xmlns="https://route53.amazonaws.com/doc/2013-04-01/">
  <Messages>
    <Message>` + errorMessage + `</Message>
  </Messages>
</InvalidChangeBatch>
`

	r := makeClientWithResponse(errorResponse)

	req := &route53.ChangeResourceRecordSetsInput{
		HostedZoneId: aws.String("zoneId"),
		ChangeBatch: &route53.ChangeBatch{
			Changes: []*route53.Change{
				{
					Action: aws.String("CREATE"),
					ResourceRecordSet: &route53.ResourceRecordSet{
						Name: aws.String("domain"),
						Type: aws.String("CNAME"),
						TTL:  aws.Int64(120),
						ResourceRecords: []*route53.ResourceRecord{
							{
								Value: aws.String("cname"),
							},
						},
					},
				},
			},
		},
	}

	_, err := r.ChangeResourceRecordSets(req)
	if err == nil {
		t.Error("expected error, but received none")
	}

	if reqErr, ok := err.(awserr.RequestFailure); ok {
		if reqErr == nil {
			t.Error("expected error, but received none")
		}

		if e, a := 400, reqErr.StatusCode(); e != a {
			t.Errorf("expected %d, but received %d", e, a)
		}
	} else {
		t.Fatal("returned error is not a RequestFailure")
	}

	if batchErr, ok := err.(awserr.BatchedErrors); ok {
		errs := batchErr.OrigErrs()
		if e, a := 1, len(errs); e != a {
			t.Errorf("expected %d, but received %d", e, a)
		}
		if e, a := "InvalidChangeBatch", errs[0].(awserr.Error).Code(); e != a {
			t.Errorf("expected %s, but received %s", e, a)
		}
		if e, a := errorMessage, errs[0].(awserr.Error).Message(); e != a {
			t.Errorf("expected %s, but received %s", e, a)
		}
	} else {
		t.Fatal("returned error is not a BatchedErrors")
	}
}
