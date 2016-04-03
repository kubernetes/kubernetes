package s3_test

import (
	"crypto/md5"
	"encoding/base64"
	"io/ioutil"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/stretchr/testify/assert"
)

func assertMD5(t *testing.T, req *request.Request) {
	err := req.Build()
	assert.NoError(t, err)

	b, _ := ioutil.ReadAll(req.HTTPRequest.Body)
	out := md5.Sum(b)
	assert.NotEmpty(t, b)
	assert.Equal(t, base64.StdEncoding.EncodeToString(out[:]), req.HTTPRequest.Header.Get("Content-MD5"))
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
