package s3_test

import (
	"net/url"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

type s3BucketTest struct {
	bucket  string
	url     string
	errCode string
}

var (
	sslTests = []s3BucketTest{
		{"abc", "https://abc.s3.mock-region.amazonaws.com/", ""},
		{"a$b$c", "https://s3.mock-region.amazonaws.com/a%24b%24c", ""},
		{"a.b.c", "https://s3.mock-region.amazonaws.com/a.b.c", ""},
		{"a..bc", "https://s3.mock-region.amazonaws.com/a..bc", ""},
	}

	nosslTests = []s3BucketTest{
		{"a.b.c", "http://a.b.c.s3.mock-region.amazonaws.com/", ""},
		{"a..bc", "http://s3.mock-region.amazonaws.com/a..bc", ""},
	}

	forcepathTests = []s3BucketTest{
		{"abc", "https://s3.mock-region.amazonaws.com/abc", ""},
		{"a$b$c", "https://s3.mock-region.amazonaws.com/a%24b%24c", ""},
		{"a.b.c", "https://s3.mock-region.amazonaws.com/a.b.c", ""},
		{"a..bc", "https://s3.mock-region.amazonaws.com/a..bc", ""},
	}

	accelerateTests = []s3BucketTest{
		{"abc", "https://abc.s3-accelerate.amazonaws.com/", ""},
		{"a.b.c", "https://s3.mock-region.amazonaws.com/%7BBucket%7D", "InvalidParameterException"},
		{"a$b$c", "https://s3.mock-region.amazonaws.com/%7BBucket%7D", "InvalidParameterException"},
	}

	accelerateNoSSLTests = []s3BucketTest{
		{"abc", "http://abc.s3-accelerate.amazonaws.com/", ""},
		{"a.b.c", "http://a.b.c.s3-accelerate.amazonaws.com/", ""},
		{"a$b$c", "http://s3.mock-region.amazonaws.com/%7BBucket%7D", "InvalidParameterException"},
	}

	accelerateDualstack = []s3BucketTest{
		{"abc", "https://abc.s3-accelerate.dualstack.amazonaws.com/", ""},
		{"a.b.c", "https://s3.dualstack.mock-region.amazonaws.com/%7BBucket%7D", "InvalidParameterException"},
		{"a$b$c", "https://s3.dualstack.mock-region.amazonaws.com/%7BBucket%7D", "InvalidParameterException"},
	}
)

func runTests(t *testing.T, svc *s3.S3, tests []s3BucketTest) {
	for i, test := range tests {
		req, _ := svc.ListObjectsRequest(&s3.ListObjectsInput{Bucket: &test.bucket})
		req.Build()
		if e, a := test.url, req.HTTPRequest.URL.String(); e != a {
			t.Errorf("%d, expect url %s, got %s", i, e, a)
		}
		if test.errCode != "" {
			if err := req.Error; err == nil {
				t.Fatalf("%d, expect no error", i)
			}
			if a, e := req.Error.(awserr.Error).Code(), test.errCode; !strings.Contains(a, e) {
				t.Errorf("%d, expect error code to contain %q, got %q", i, e, a)
			}
		}
	}
}

func TestAccelerateBucketBuild(t *testing.T) {
	s := s3.New(unit.Session, &aws.Config{S3UseAccelerate: aws.Bool(true)})
	runTests(t, s, accelerateTests)
}

func TestAccelerateNoSSLBucketBuild(t *testing.T) {
	s := s3.New(unit.Session, &aws.Config{S3UseAccelerate: aws.Bool(true), DisableSSL: aws.Bool(true)})
	runTests(t, s, accelerateNoSSLTests)
}

func TestAccelerateDualstackBucketBuild(t *testing.T) {
	s := s3.New(unit.Session, &aws.Config{
		S3UseAccelerate: aws.Bool(true),
		UseDualStack:    aws.Bool(true),
	})
	runTests(t, s, accelerateDualstack)
}

func TestHostStyleBucketBuild(t *testing.T) {
	s := s3.New(unit.Session)
	runTests(t, s, sslTests)
}

func TestHostStyleBucketBuildNoSSL(t *testing.T) {
	s := s3.New(unit.Session, &aws.Config{DisableSSL: aws.Bool(true)})
	runTests(t, s, nosslTests)
}

func TestPathStyleBucketBuild(t *testing.T) {
	s := s3.New(unit.Session, &aws.Config{S3ForcePathStyle: aws.Bool(true)})
	runTests(t, s, forcepathTests)
}

func TestHostStyleBucketGetBucketLocation(t *testing.T) {
	s := s3.New(unit.Session)
	req, _ := s.GetBucketLocationRequest(&s3.GetBucketLocationInput{
		Bucket: aws.String("bucket"),
	})

	req.Build()
	if req.Error != nil {
		t.Fatalf("expect no error, got %v", req.Error)
	}
	u, _ := url.Parse(req.HTTPRequest.URL.String())
	if e, a := "bucket", u.Host; strings.Contains(a, e) {
		t.Errorf("expect %s to not be in %s", e, a)
	}
	if e, a := "bucket", u.Path; !strings.Contains(a, e) {
		t.Errorf("expect %s to be in %s", e, a)
	}
}
