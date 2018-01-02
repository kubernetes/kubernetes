package s3_test

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

var s3LocationTests = []struct {
	body string
	loc  string
}{
	{`<?xml version="1.0" encoding="UTF-8"?><LocationConstraint xmlns="http://s3.amazonaws.com/doc/2006-03-01/"/>`, ``},
	{`<?xml version="1.0" encoding="UTF-8"?><LocationConstraint xmlns="http://s3.amazonaws.com/doc/2006-03-01/">EU</LocationConstraint>`, `EU`},
}

func TestGetBucketLocation(t *testing.T) {
	for _, test := range s3LocationTests {
		s := s3.New(unit.Session)
		s.Handlers.Send.Clear()
		s.Handlers.Send.PushBack(func(r *request.Request) {
			reader := ioutil.NopCloser(bytes.NewReader([]byte(test.body)))
			r.HTTPResponse = &http.Response{StatusCode: 200, Body: reader}
		})

		resp, err := s.GetBucketLocation(&s3.GetBucketLocationInput{Bucket: aws.String("bucket")})
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		if test.loc == "" {
			if v := resp.LocationConstraint; v != nil {
				t.Errorf("expect location constraint to be nil, got %s", *v)
			}
		} else {
			if e, a := test.loc, *resp.LocationConstraint; e != a {
				t.Errorf("expect %s location constraint, got %v", e, a)
			}
		}
	}
}

func TestNormalizeBucketLocation(t *testing.T) {
	cases := []struct {
		In, Out string
	}{
		{"", "us-east-1"},
		{"EU", "eu-west-1"},
		{"us-east-1", "us-east-1"},
		{"something", "something"},
	}

	for i, c := range cases {
		actual := s3.NormalizeBucketLocation(c.In)
		if e, a := c.Out, actual; e != a {
			t.Errorf("%d, expect %s bucket location, got %s", i, e, a)
		}
	}
}

func TestWithNormalizeBucketLocation(t *testing.T) {
	req := &request.Request{}
	req.ApplyOptions(s3.WithNormalizeBucketLocation)

	cases := []struct {
		In, Out string
	}{
		{"", "us-east-1"},
		{"EU", "eu-west-1"},
		{"us-east-1", "us-east-1"},
		{"something", "something"},
	}

	for i, c := range cases {
		req.Data = &s3.GetBucketLocationOutput{
			LocationConstraint: aws.String(c.In),
		}
		req.Handlers.Unmarshal.Run(req)

		v := req.Data.(*s3.GetBucketLocationOutput).LocationConstraint
		if e, a := c.Out, aws.StringValue(v); e != a {
			t.Errorf("%d, expect %s bucket location, got %s", i, e, a)
		}
	}
}

func TestPopulateLocationConstraint(t *testing.T) {
	s := s3.New(unit.Session)
	in := &s3.CreateBucketInput{
		Bucket: aws.String("bucket"),
	}
	req, _ := s.CreateBucketRequest(in)
	if err := req.Build(); err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	v, _ := awsutil.ValuesAtPath(req.Params, "CreateBucketConfiguration.LocationConstraint")
	if e, a := "mock-region", *(v[0].(*string)); e != a {
		t.Errorf("expect %s location constraint, got %s", e, a)
	}
	if v := in.CreateBucketConfiguration; v != nil {
		// don't modify original params
		t.Errorf("expect create bucket Configuration to be nil, got %s", *v)
	}
}

func TestNoPopulateLocationConstraintIfProvided(t *testing.T) {
	s := s3.New(unit.Session)
	req, _ := s.CreateBucketRequest(&s3.CreateBucketInput{
		Bucket: aws.String("bucket"),
		CreateBucketConfiguration: &s3.CreateBucketConfiguration{},
	})
	if err := req.Build(); err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	v, _ := awsutil.ValuesAtPath(req.Params, "CreateBucketConfiguration.LocationConstraint")
	if l := len(v); l != 0 {
		t.Errorf("expect no values, got %d", l)
	}
}

func TestNoPopulateLocationConstraintIfClassic(t *testing.T) {
	s := s3.New(unit.Session, &aws.Config{Region: aws.String("us-east-1")})
	req, _ := s.CreateBucketRequest(&s3.CreateBucketInput{
		Bucket: aws.String("bucket"),
	})
	if err := req.Build(); err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	v, _ := awsutil.ValuesAtPath(req.Params, "CreateBucketConfiguration.LocationConstraint")
	if l := len(v); l != 0 {
		t.Errorf("expect no values, got %d", l)
	}
}
