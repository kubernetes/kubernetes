// +build go1.7

package ec2_test

import (
	"bytes"
	"context"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	sdkclient "github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/ec2"
)

func TestCopySnapshotPresignedURL(t *testing.T) {
	svc := ec2.New(unit.Session, &aws.Config{Region: aws.String("us-west-2")})

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("expect CopySnapshotRequest with nill")
			}
		}()
		// Doesn't panic on nil input
		req, _ := svc.CopySnapshotRequest(nil)
		req.Sign()
	}()

	req, _ := svc.CopySnapshotRequest(&ec2.CopySnapshotInput{
		SourceRegion:     aws.String("us-west-1"),
		SourceSnapshotId: aws.String("snap-id"),
	})
	req.Sign()

	b, _ := ioutil.ReadAll(req.HTTPRequest.Body)
	q, _ := url.ParseQuery(string(b))
	u, _ := url.QueryUnescape(q.Get("PresignedUrl"))
	if e, a := "us-west-2", q.Get("DestinationRegion"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "us-west-1", q.Get("SourceRegion"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	r := regexp.MustCompile(`^https://ec2\.us-west-1\.amazonaws\.com/.+&DestinationRegion=us-west-2`)
	if !r.MatchString(u) {
		t.Errorf("expect %v to match, got %v", r.String(), u)
	}
}

func TestNoCustomRetryerWithMaxRetries(t *testing.T) {
	cases := map[string]struct {
		Config           aws.Config
		ExpectMaxRetries int
	}{
		"With custom retrier": {
			Config: aws.Config{
				Retryer: sdkclient.DefaultRetryer{
					NumMaxRetries: 10,
				},
			},
			ExpectMaxRetries: 10,
		},
		"with max retries": {
			Config: aws.Config{
				MaxRetries: aws.Int(10),
			},
			ExpectMaxRetries: 10,
		},
		"no options set": {
			ExpectMaxRetries: sdkclient.DefaultRetryerMaxNumRetries,
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			client := ec2.New(unit.Session, &aws.Config{
				DisableParamValidation: aws.Bool(true),
			}, c.Config.Copy())
			client.ModifyNetworkInterfaceAttributeWithContext(context.Background(), nil, checkRetryerMaxRetries(t, c.ExpectMaxRetries))
			client.AssignPrivateIpAddressesWithContext(context.Background(), nil, checkRetryerMaxRetries(t, c.ExpectMaxRetries))
		})
	}

}

func checkRetryerMaxRetries(t *testing.T, maxRetries int) func(*request.Request) {
	return func(r *request.Request) {
		r.Handlers.Send.Clear()
		r.Handlers.Send.PushBack(func(rr *request.Request) {
			if e, a := maxRetries, rr.Retryer.MaxRetries(); e != a {
				t.Errorf("%s, expect %v max retries, got %v", rr.Operation.Name, e, a)
			}
			rr.HTTPResponse = &http.Response{
				StatusCode: 200,
				Header:     http.Header{},
				Body:       ioutil.NopCloser(&bytes.Buffer{}),
			}
		})
	}
}
