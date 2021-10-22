package s3manager

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

func testSetupGetBucketRegionServer(region string, statusCode int, incHeader bool) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if incHeader {
			w.Header().Set(bucketRegionHeader, region)
		}
		w.WriteHeader(statusCode)
	}))
}

var testGetBucketRegionCases = []struct {
	RespRegion      string
	StatusCode      int
	HintRegion      string
	ExpectReqRegion string
}{
	{"bucket-region", 301, "hint-region", ""},
	{"bucket-region", 403, "hint-region", ""},
	{"bucket-region", 200, "hint-region", ""},
	{"bucket-region", 200, "", "default-region"},
}

func TestGetBucketRegion_Exists(t *testing.T) {
	for i, c := range testGetBucketRegionCases {
		server := testSetupGetBucketRegionServer(c.RespRegion, c.StatusCode, true)
		defer server.Close()

		sess := unit.Session.Copy()
		sess.Config.Region = aws.String("default-region")
		sess.Config.Endpoint = aws.String(server.URL)
		sess.Config.DisableSSL = aws.Bool(true)

		ctx := aws.BackgroundContext()
		region, err := GetBucketRegion(ctx, sess, "bucket", c.HintRegion)
		if err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}
		if e, a := c.RespRegion, region; e != a {
			t.Errorf("%d, expect %q region, got %q", i, e, a)
		}
	}
}

func TestGetBucketRegion_NotExists(t *testing.T) {
	server := testSetupGetBucketRegionServer("ignore-region", 404, false)
	defer server.Close()

	sess := unit.Session.Copy()
	sess.Config.Endpoint = aws.String(server.URL)
	sess.Config.DisableSSL = aws.Bool(true)

	ctx := aws.BackgroundContext()
	region, err := GetBucketRegion(ctx, sess, "bucket", "hint-region")
	if err == nil {
		t.Fatalf("expect error, but did not get one")
	}
	aerr := err.(awserr.Error)
	if e, a := "NotFound", aerr.Code(); e != a {
		t.Errorf("expect %s error code, got %s", e, a)
	}
	if len(region) != 0 {
		t.Errorf("expect region not to be set, got %q", region)
	}
}

func TestGetBucketRegionWithClient(t *testing.T) {
	for i, c := range testGetBucketRegionCases {
		server := testSetupGetBucketRegionServer(c.RespRegion, c.StatusCode, true)
		defer server.Close()

		svc := s3.New(unit.Session, &aws.Config{
			Region:     aws.String("hint-region"),
			Endpoint:   aws.String(server.URL),
			DisableSSL: aws.Bool(true),
		})

		ctx := aws.BackgroundContext()

		region, err := GetBucketRegionWithClient(ctx, svc, "bucket")
		if err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}
		if e, a := c.RespRegion, region; e != a {
			t.Errorf("%d, expect %q region, got %q", i, e, a)
		}
	}
}

func TestGetBucketRegionWithClientWithoutRegion(t *testing.T) {
	for i, c := range testGetBucketRegionCases {
		server := testSetupGetBucketRegionServer(c.RespRegion, c.StatusCode, true)
		defer server.Close()

		svc := s3.New(unit.Session, &aws.Config{
			Endpoint:   aws.String(server.URL),
			DisableSSL: aws.Bool(true),
		})

		ctx := aws.BackgroundContext()

		region, err := GetBucketRegionWithClient(ctx, svc, "bucket")
		if err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}
		if e, a := c.RespRegion, region; e != a {
			t.Errorf("%d, expect %q region, got %q", i, e, a)
		}
	}
}
