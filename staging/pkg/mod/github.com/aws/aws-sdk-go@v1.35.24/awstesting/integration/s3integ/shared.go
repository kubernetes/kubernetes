// +build integration

package s3integ

import (
	"fmt"

	"github.com/aws/aws-sdk-go/awstesting/integration"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
	"github.com/aws/aws-sdk-go/service/s3control"
	"github.com/aws/aws-sdk-go/service/s3control/s3controliface"
)

// BucketPrefix is the root prefix of integration test buckets.
const BucketPrefix = "aws-sdk-go-integration"

// GenerateBucketName returns a unique bucket name.
func GenerateBucketName() string {
	return fmt.Sprintf("%s-%s",
		BucketPrefix, integration.UniqueID())
}

// SetupBucket returns a test bucket created for the integration tests.
func SetupBucket(svc s3iface.S3API, bucketName string) (err error) {

	fmt.Println("Setup: Creating test bucket,", bucketName)
	_, err = svc.CreateBucket(&s3.CreateBucketInput{Bucket: &bucketName})
	if err != nil {
		return fmt.Errorf("failed to create bucket %s, %v", bucketName, err)
	}

	fmt.Println("Setup: Waiting for bucket to exist,", bucketName)
	err = svc.WaitUntilBucketExists(&s3.HeadBucketInput{Bucket: &bucketName})
	if err != nil {
		return fmt.Errorf("failed waiting for bucket %s to be created, %v",
			bucketName, err)
	}

	return nil
}

// CleanupBucket deletes the contents of a S3 bucket, before deleting the bucket
// it self.
func CleanupBucket(svc s3iface.S3API, bucketName string) error {
	errs := []error{}

	fmt.Println("TearDown: Deleting objects from test bucket,", bucketName)
	err := svc.ListObjectsPages(
		&s3.ListObjectsInput{Bucket: &bucketName},
		func(page *s3.ListObjectsOutput, lastPage bool) bool {
			for _, o := range page.Contents {
				_, err := svc.DeleteObject(&s3.DeleteObjectInput{
					Bucket: &bucketName,
					Key:    o.Key,
				})
				if err != nil {
					errs = append(errs, err)
				}
			}
			return true
		},
	)
	if err != nil {
		return fmt.Errorf("failed to list objects, %s, %v", bucketName, err)
	}

	fmt.Println("TearDown: Deleting partial uploads from test bucket,", bucketName)
	err = svc.ListMultipartUploadsPages(
		&s3.ListMultipartUploadsInput{Bucket: &bucketName},
		func(page *s3.ListMultipartUploadsOutput, lastPage bool) bool {
			for _, u := range page.Uploads {
				svc.AbortMultipartUpload(&s3.AbortMultipartUploadInput{
					Bucket:   &bucketName,
					Key:      u.Key,
					UploadId: u.UploadId,
				})
			}
			return true
		},
	)
	if err != nil {
		return fmt.Errorf("failed to list multipart objects, %s, %v", bucketName, err)
	}

	if len(errs) != 0 {
		return fmt.Errorf("failed to delete objects, %s", errs)
	}

	fmt.Println("TearDown: Deleting test bucket,", bucketName)
	if _, err = svc.DeleteBucket(&s3.DeleteBucketInput{Bucket: &bucketName}); err != nil {
		return fmt.Errorf("failed to delete test bucket, %s", bucketName)
	}

	return nil
}

// SetupAccessPoint returns an access point for the given bucket for testing
func SetupAccessPoint(svc s3controliface.S3ControlAPI, account, bucket, accessPoint string) error {
	fmt.Printf("Setup: creating access point %q for bucket %q\n", accessPoint, bucket)
	_, err := svc.CreateAccessPoint(&s3control.CreateAccessPointInput{
		AccountId: &account,
		Bucket:    &bucket,
		Name:      &accessPoint,
	})
	if err != nil {
		return fmt.Errorf("failed to create access point: %v", err)
	}
	return nil
}

// CleanupAccessPoint deletes the given access point
func CleanupAccessPoint(svc s3controliface.S3ControlAPI, account, accessPoint string) error {
	fmt.Printf("TearDown: Deleting access point %q\n", accessPoint)
	_, err := svc.DeleteAccessPoint(&s3control.DeleteAccessPointInput{
		AccountId: &account,
		Name:      &accessPoint,
	})
	if err != nil {
		return fmt.Errorf("failed to delete access point: %v", err)
	}
	return nil
}
