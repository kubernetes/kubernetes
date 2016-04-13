package s3_test

import (
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

type s3BucketTest struct {
	bucket string
	url    string
}

var (
	sslTests = []s3BucketTest{
		{"abc", "https://abc.s3.mock-region.amazonaws.com/"},
		{"a$b$c", "https://s3.mock-region.amazonaws.com/a%24b%24c"},
		{"a.b.c", "https://s3.mock-region.amazonaws.com/a.b.c"},
		{"a..bc", "https://s3.mock-region.amazonaws.com/a..bc"},
	}

	nosslTests = []s3BucketTest{
		{"a.b.c", "http://a.b.c.s3.mock-region.amazonaws.com/"},
		{"a..bc", "http://s3.mock-region.amazonaws.com/a..bc"},
	}

	forcepathTests = []s3BucketTest{
		{"abc", "https://s3.mock-region.amazonaws.com/abc"},
		{"a$b$c", "https://s3.mock-region.amazonaws.com/a%24b%24c"},
		{"a.b.c", "https://s3.mock-region.amazonaws.com/a.b.c"},
		{"a..bc", "https://s3.mock-region.amazonaws.com/a..bc"},
	}
)

func runTests(t *testing.T, svc *s3.S3, tests []s3BucketTest) {
	for _, test := range tests {
		req, _ := svc.ListObjectsRequest(&s3.ListObjectsInput{Bucket: &test.bucket})
		req.Build()
		assert.Equal(t, test.url, req.HTTPRequest.URL.String())
	}
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
	require.NoError(t, req.Error)
	u, _ := url.Parse(req.HTTPRequest.URL.String())
	assert.NotContains(t, u.Host, "bucket")
	assert.Contains(t, u.Path, "bucket")
}
