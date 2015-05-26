// +build !integration

package ec2_test

import (
	"io/ioutil"
	"net/url"
	"testing"

	"github.com/awslabs/aws-sdk-go/aws"
	"github.com/awslabs/aws-sdk-go/internal/test/unit"
	"github.com/awslabs/aws-sdk-go/service/ec2"
	"github.com/stretchr/testify/assert"
)

var _ = unit.Imported

func TestCopySnapshotPresignedURL(t *testing.T) {
	svc := ec2.New(&aws.Config{Region: "us-west-2"})

	assert.NotPanics(t, func() {
		// Doesn't panic on nil input
		req, _ := svc.CopySnapshotRequest(nil)
		req.Sign()
	})

	req, _ := svc.CopySnapshotRequest(&ec2.CopySnapshotInput{
		SourceRegion:     aws.String("us-west-1"),
		SourceSnapshotID: aws.String("snap-id"),
	})
	req.Sign()

	b, _ := ioutil.ReadAll(req.HTTPRequest.Body)
	q, _ := url.ParseQuery(string(b))
	url, _ := url.QueryUnescape(q.Get("PresignedUrl"))
	assert.Equal(t, "us-west-2", q.Get("DestinationRegion"))
	assert.Regexp(t, `^https://ec2\.us-west-1\.amazon.+&DestinationRegion=us-west-2`, url)
}
