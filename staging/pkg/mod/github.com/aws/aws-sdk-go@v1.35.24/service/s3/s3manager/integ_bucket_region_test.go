// +build integration

package s3manager_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

func TestGetBucketRegion(t *testing.T) {
	expectRegion := aws.StringValue(integSess.Config.Region)

	ctx := aws.BackgroundContext()
	region, err := s3manager.GetBucketRegion(ctx, integSess,
		aws.StringValue(bucketName), expectRegion)

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	if e, a := expectRegion, region; e != a {
		t.Errorf("expect %s bucket region, got %s", e, a)
	}
}
