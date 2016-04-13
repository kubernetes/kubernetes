package ec2_test

import (
	"io/ioutil"
	"net/url"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/stretchr/testify/assert"
)

func TestCopySnapshotPresignedURL(t *testing.T) {
	svc := ec2.New(unit.Session, &aws.Config{Region: aws.String("us-west-2")})

	assert.NotPanics(t, func() {
		// Doesn't panic on nil input
		req, _ := svc.CopySnapshotRequest(nil)
		req.Sign()
	})

	req, _ := svc.CopySnapshotRequest(&ec2.CopySnapshotInput{
		SourceRegion:     aws.String("us-west-1"),
		SourceSnapshotId: aws.String("snap-id"),
	})
	req.Sign()

	b, _ := ioutil.ReadAll(req.HTTPRequest.Body)
	q, _ := url.ParseQuery(string(b))
	u, _ := url.QueryUnescape(q.Get("PresignedUrl"))
	assert.Equal(t, "us-west-2", q.Get("DestinationRegion"))
	assert.Equal(t, "us-west-1", q.Get("SourceRegion"))
	assert.Regexp(t, `^https://ec2\.us-west-1\.amazonaws\.com/.+&DestinationRegion=us-west-2`, u)
}
