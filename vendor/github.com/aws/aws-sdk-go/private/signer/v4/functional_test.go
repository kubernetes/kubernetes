package v4_test

import (
	"net/url"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

func TestPresignHandler(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket:             aws.String("bucket"),
		Key:                aws.String("key"),
		ContentDisposition: aws.String("a+b c$d"),
		ACL:                aws.String("public-read"),
	})
	req.Time = time.Unix(0, 0)
	urlstr, err := req.Presign(5 * time.Minute)

	assert.NoError(t, err)

	expectedDate := "19700101T000000Z"
	expectedHeaders := "host;x-amz-acl"
	expectedSig := "7edcb4e3a1bf12f4989018d75acbe3a7f03df24bd6f3112602d59fc551f0e4e2"
	expectedCred := "AKID/19700101/mock-region/s3/aws4_request"

	u, _ := url.Parse(urlstr)
	urlQ := u.Query()
	assert.Equal(t, expectedSig, urlQ.Get("X-Amz-Signature"))
	assert.Equal(t, expectedCred, urlQ.Get("X-Amz-Credential"))
	assert.Equal(t, expectedHeaders, urlQ.Get("X-Amz-SignedHeaders"))
	assert.Equal(t, expectedDate, urlQ.Get("X-Amz-Date"))
	assert.Equal(t, "300", urlQ.Get("X-Amz-Expires"))

	assert.NotContains(t, urlstr, "+") // + encoded as %20
}
