package s3manager

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

// GetBucketRegion will attempt to get the region for a bucket using the
// regionHint to determine which AWS partition to perform the query on.
//
// The request will not be signed, and will not use your AWS credentials.
//
// A "NotFound" error code will be returned if the bucket does not exist in
// the AWS partition the regionHint belongs to.
//
// For example to get the region of a bucket which exists in "eu-central-1"
// you could provide a region hint of "us-west-2".
//
//    sess := session.Must(session.NewSession())
//
//    bucket := "my-bucket"
//    region, err := s3manager.GetBucketRegion(ctx, sess, bucket, "us-west-2")
//    if err != nil {
//        if aerr, ok := err.(awserr.Error); ok && aerr.Code() == "NotFound" {
//             fmt.Fprintf(os.Stderr, "unable to find bucket %s's region not found\n", bucket)
//        }
//        return err
//    }
//    fmt.Printf("Bucket %s is in %s region\n", bucket, region)
//
func GetBucketRegion(ctx aws.Context, c client.ConfigProvider, bucket, regionHint string, opts ...request.Option) (string, error) {
	svc := s3.New(c, &aws.Config{
		Region: aws.String(regionHint),
	})
	return GetBucketRegionWithClient(ctx, svc, bucket, opts...)
}

const bucketRegionHeader = "X-Amz-Bucket-Region"

// GetBucketRegionWithClient is the same as GetBucketRegion with the exception
// that it takes a S3 service client instead of a Session. The regionHint is
// derived from the region the S3 service client was created in.
//
// See GetBucketRegion for more information.
func GetBucketRegionWithClient(ctx aws.Context, svc s3iface.S3API, bucket string, opts ...request.Option) (string, error) {
	req, _ := svc.HeadBucketRequest(&s3.HeadBucketInput{
		Bucket: aws.String(bucket),
	})
	req.Config.S3ForcePathStyle = aws.Bool(true)
	req.Config.Credentials = credentials.AnonymousCredentials
	req.SetContext(ctx)

	// Disable HTTP redirects to prevent an invalid 301 from eating the response
	// because Go's HTTP client will fail, and drop the response if an 301 is
	// received without a location header. S3 will return a 301 without the
	// location header for HeadObject API calls.
	req.DisableFollowRedirects = true

	var bucketRegion string
	req.Handlers.Send.PushBack(func(r *request.Request) {
		bucketRegion = r.HTTPResponse.Header.Get(bucketRegionHeader)
		if len(bucketRegion) == 0 {
			return
		}
		r.HTTPResponse.StatusCode = 200
		r.HTTPResponse.Status = "OK"
		r.Error = nil
	})

	req.ApplyOptions(opts...)

	if err := req.Send(); err != nil {
		return "", err
	}

	bucketRegion = s3.NormalizeBucketLocation(bucketRegion)

	return bucketRegion, nil
}
