package s3manager

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
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
// A "NotFound" error code will be returned if the bucket does not exist in the
// AWS partition the regionHint belongs to. If the regionHint parameter is an
// empty string GetBucketRegion will fallback to the ConfigProvider's region
// config. If the regionHint is empty, and the ConfigProvider does not have a
// region value, an error will be returned..
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
// By default the request will be made to the Amazon S3 endpoint using the Path
// style addressing.
//
//    s3.us-west-2.amazonaws.com/bucketname
//
// This is not compatible with Amazon S3's FIPS endpoints. To override this
// behavior to use Virtual Host style addressing, provide a functional option
// that will set the Request's Config.S3ForcePathStyle to aws.Bool(false).
//
//    region, err := s3manager.GetBucketRegion(ctx, sess, "bucketname", "us-west-2", func(r *request.Request) {
//        r.S3ForcePathStyle = aws.Bool(false)
//    })
//
// To configure the GetBucketRegion to make a request via the Amazon
// S3 FIPS endpoints directly when a FIPS region name is not available, (e.g.
// fips-us-gov-west-1) set the Config.Endpoint on the Session, or client the
// utility is called with. The hint region will be ignored if an endpoint URL
// is configured on the session or client.
//
//    sess, err := session.NewSession(&aws.Config{
//        Endpoint: aws.String("https://s3-fips.us-west-2.amazonaws.com"),
//    })
//
//    region, err := s3manager.GetBucketRegion(context.Background(), sess, "bucketname", "")
func GetBucketRegion(ctx aws.Context, c client.ConfigProvider, bucket, regionHint string, opts ...request.Option) (string, error) {
	var cfg aws.Config
	if len(regionHint) != 0 {
		cfg.Region = aws.String(regionHint)
	}
	svc := s3.New(c, &cfg)
	return GetBucketRegionWithClient(ctx, svc, bucket, opts...)
}

const bucketRegionHeader = "X-Amz-Bucket-Region"

// GetBucketRegionWithClient is the same as GetBucketRegion with the exception
// that it takes a S3 service client instead of a Session. The regionHint is
// derived from the region the S3 service client was created in.
//
// By default the request will be made to the Amazon S3 endpoint using the Path
// style addressing.
//
//    s3.us-west-2.amazonaws.com/bucketname
//
// This is not compatible with Amazon S3's FIPS endpoints. To override this
// behavior to use Virtual Host style addressing, provide a functional option
// that will set the Request's Config.S3ForcePathStyle to aws.Bool(false).
//
//    region, err := s3manager.GetBucketRegionWithClient(ctx, client, "bucketname", func(r *request.Request) {
//        r.S3ForcePathStyle = aws.Bool(false)
//    })
//
// To configure the GetBucketRegion to make a request via the Amazon
// S3 FIPS endpoints directly when a FIPS region name is not available, (e.g.
// fips-us-gov-west-1) set the Config.Endpoint on the Session, or client the
// utility is called with. The hint region will be ignored if an endpoint URL
// is configured on the session or client.
//
//    region, err := s3manager.GetBucketRegionWithClient(context.Background(),
//    s3.New(sess, &aws.Config{
//        Endpoint: aws.String("https://s3-fips.us-west-2.amazonaws.com"),
//    }),
//    "bucketname")
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
	// Replace the endpoint validation handler to not require a region if an
	// endpoint URL was specified. Since these requests are not authenticated,
	// requiring a region is not needed when an endpoint URL is provided.
	req.Handlers.Validate.Swap(
		corehandlers.ValidateEndpointHandler.Name,
		request.NamedHandler{
			Name: "validateEndpointWithoutRegion",
			Fn:   validateEndpointWithoutRegion,
		},
	)

	req.ApplyOptions(opts...)

	if err := req.Send(); err != nil {
		return "", err
	}

	bucketRegion = s3.NormalizeBucketLocation(bucketRegion)

	return bucketRegion, nil
}

func validateEndpointWithoutRegion(r *request.Request) {
	// Check if the caller provided an explicit URL instead of one derived by
	// the SDK's endpoint resolver. For GetBucketRegion, with an explicit
	// endpoint URL, a region is not needed. If no endpoint URL is provided,
	// fallback the SDK's standard endpoint validation handler.
	if len(aws.StringValue(r.Config.Endpoint)) == 0 {
		corehandlers.ValidateEndpointHandler.Fn(r)
	}
}
