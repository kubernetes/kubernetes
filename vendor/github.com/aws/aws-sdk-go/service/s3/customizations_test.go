package s3_test

import (
	"crypto/md5"
	"encoding/base64"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

func assertMD5(t *testing.T, req *request.Request) {
	err := req.Build()
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	b, _ := ioutil.ReadAll(req.HTTPRequest.Body)
	out := md5.Sum(b)
	if len(b) == 0 {
		t.Error("expected non-empty value")
	}
	if e, a := base64.StdEncoding.EncodeToString(out[:]), req.HTTPRequest.Header.Get("Content-MD5"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
}

func TestMD5InPutBucketCors(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketCorsRequest(&s3.PutBucketCorsInput{
		Bucket: aws.String("bucketname"),
		CORSConfiguration: &s3.CORSConfiguration{
			CORSRules: []*s3.CORSRule{
				{
					AllowedMethods: []*string{aws.String("GET")},
					AllowedOrigins: []*string{aws.String("*")},
				},
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketLifecycle(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketLifecycleRequest(&s3.PutBucketLifecycleInput{
		Bucket: aws.String("bucketname"),
		LifecycleConfiguration: &s3.LifecycleConfiguration{
			Rules: []*s3.Rule{
				{
					ID:     aws.String("ID"),
					Prefix: aws.String("Prefix"),
					Status: aws.String("Enabled"),
				},
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketPolicy(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketPolicyRequest(&s3.PutBucketPolicyInput{
		Bucket: aws.String("bucketname"),
		Policy: aws.String("{}"),
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketTagging(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketTaggingRequest(&s3.PutBucketTaggingInput{
		Bucket: aws.String("bucketname"),
		Tagging: &s3.Tagging{
			TagSet: []*s3.Tag{
				{Key: aws.String("KEY"), Value: aws.String("VALUE")},
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InDeleteObjects(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.DeleteObjectsRequest(&s3.DeleteObjectsInput{
		Bucket: aws.String("bucketname"),
		Delete: &s3.Delete{
			Objects: []*s3.ObjectIdentifier{
				{Key: aws.String("key")},
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketLifecycleConfiguration(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketLifecycleConfigurationRequest(&s3.PutBucketLifecycleConfigurationInput{
		Bucket: aws.String("bucketname"),
		LifecycleConfiguration: &s3.BucketLifecycleConfiguration{
			Rules: []*s3.LifecycleRule{
				{Prefix: aws.String("prefix"), Status: aws.String(s3.ExpirationStatusEnabled)},
			},
		},
	})
	assertMD5(t, req)
}

const (
	metaKeyPrefix = `X-Amz-Meta-`
	utf8KeySuffix = `My-Info`
	utf8Value     = "hello-世界\u0444"
)

func TestPutObjectMetadataWithUnicode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if e, a := utf8Value, r.Header.Get(metaKeyPrefix+utf8KeySuffix); e != a {
			t.Errorf("expected %s, but received %s", e, a)
		}
	}))
	svc := s3.New(unit.Session, &aws.Config{
		Endpoint:   aws.String(server.URL),
		DisableSSL: aws.Bool(true),
	})

	_, err := svc.PutObject(&s3.PutObjectInput{
		Bucket: aws.String("my_bucket"),
		Key:    aws.String("my_key"),
		Body:   strings.NewReader(""),
		Metadata: func() map[string]*string {
			v := map[string]*string{}
			v[utf8KeySuffix] = aws.String(utf8Value)
			return v
		}(),
	})

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
}

func TestGetObjectMetadataWithUnicode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(metaKeyPrefix+utf8KeySuffix, utf8Value)
	}))
	svc := s3.New(unit.Session, &aws.Config{
		Endpoint:   aws.String(server.URL),
		DisableSSL: aws.Bool(true),
	})

	resp, err := svc.GetObject(&s3.GetObjectInput{
		Bucket: aws.String("my_bucket"),
		Key:    aws.String("my_key"),
	})

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	resp.Body.Close()

	if e, a := utf8Value, *resp.Metadata[utf8KeySuffix]; e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}

}
