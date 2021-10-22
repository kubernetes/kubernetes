// +build integration

package s3_test

import (
	"bytes"
	"context"
	"crypto/tls"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/arn"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/integration"
	"github.com/aws/aws-sdk-go/awstesting/integration/s3integ"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3control"
	"github.com/aws/aws-sdk-go/service/sts"
)

const integBucketPrefix = "aws-sdk-go-integration"

var integMetadata = struct {
	AccountID string
	Region    string
	Buckets   struct {
		Source struct {
			Name string
			ARN  string
		}
		Target struct {
			Name string
			ARN  string
		}
	}

	AccessPoints struct {
		Source struct {
			Name string
			ARN  string
		}
		Target struct {
			Name string
			ARN  string
		}
	}
}{}

var s3Svc *s3.S3
var s3ControlSvc *s3control.S3Control
var stsSvc *sts.STS
var httpClient *http.Client

// TODO: (Westeros) Remove Custom Resolver Usage Before Launch
type customS3Resolver struct {
	endpoint string
	withTLS  bool
	region   string
}

func (r customS3Resolver) EndpointFor(service, _ string, opts ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
	switch strings.ToLower(service) {
	case "s3-control":
	case "s3":
	default:
		return endpoints.ResolvedEndpoint{}, fmt.Errorf("unsupported in custom resolver")
	}

	return endpoints.ResolvedEndpoint{
		PartitionID:   "aws",
		SigningRegion: r.region,
		SigningName:   "s3",
		SigningMethod: "s3v4",
		URL:           endpoints.AddScheme(r.endpoint, r.withTLS),
	}, nil
}

func TestMain(m *testing.M) {
	var result int
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintln(os.Stderr, "S3 integration tests paniced,", r)
			result = 1
		}
		os.Exit(result)
	}()

	var verifyTLS bool
	var s3Endpoint, s3ControlEndpoint string
	var s3EnableTLS, s3ControlEnableTLS bool

	flag.StringVar(&s3Endpoint, "s3-endpoint", "", "integration endpoint for S3")
	flag.BoolVar(&s3EnableTLS, "s3-tls", true, "enable TLS for S3 endpoint")

	flag.StringVar(&s3ControlEndpoint, "s3-control-endpoint", "", "integration endpoint for S3")
	flag.BoolVar(&s3ControlEnableTLS, "s3-control-tls", true, "enable TLS for S3 control endpoint")

	flag.StringVar(&integMetadata.AccountID, "account", "", "integration account id")
	flag.BoolVar(&verifyTLS, "verify-tls", true, "verify server TLS certificate")
	flag.Parse()

	httpClient = &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: verifyTLS},
		}}

	sess := integration.SessionWithDefaultRegion("us-west-2").Copy(&aws.Config{
		HTTPClient: httpClient,
	})

	var s3EndpointResolver endpoints.Resolver
	if len(s3Endpoint) != 0 {
		s3EndpointResolver = customS3Resolver{
			endpoint: s3Endpoint,
			withTLS:  s3EnableTLS,
			region:   aws.StringValue(sess.Config.Region),
		}
	}
	s3Svc = s3.New(sess, &aws.Config{
		DisableSSL:       aws.Bool(!s3EnableTLS),
		EndpointResolver: s3EndpointResolver,
	})

	var s3ControlEndpointResolver endpoints.Resolver
	if len(s3Endpoint) != 0 {
		s3ControlEndpointResolver = customS3Resolver{
			endpoint: s3ControlEndpoint,
			withTLS:  s3ControlEnableTLS,
			region:   aws.StringValue(sess.Config.Region),
		}
	}
	s3ControlSvc = s3control.New(sess, &aws.Config{
		DisableSSL:       aws.Bool(!s3ControlEnableTLS),
		EndpointResolver: s3ControlEndpointResolver,
	})
	stsSvc = sts.New(sess)

	var err error
	integMetadata.AccountID, err = getAccountID()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to get integration aws account id: %v\n", err)
		result = 1
		return
	}

	bucketCleanup, err := setupBuckets()
	defer bucketCleanup()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to setup integration test buckets: %v\n", err)
		result = 1
		return
	}

	accessPointsCleanup, err := setupAccessPoints()
	defer accessPointsCleanup()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to setup integration test access points: %v\n", err)
		result = 1
		return
	}

	result = m.Run()
}

func getAccountID() (string, error) {
	if len(integMetadata.AccountID) != 0 {
		return integMetadata.AccountID, nil
	}

	output, err := stsSvc.GetCallerIdentity(nil)
	if err != nil {
		return "", fmt.Errorf("faield to get sts caller identity")
	}

	return *output.Account, nil
}

func setupBuckets() (func(), error) {
	var cleanups []func()

	cleanup := func() {
		for i := range cleanups {
			cleanups[i]()
		}
	}

	bucketCreates := []struct {
		name *string
		arn  *string
	}{
		{name: &integMetadata.Buckets.Source.Name, arn: &integMetadata.Buckets.Source.ARN},
		{name: &integMetadata.Buckets.Target.Name, arn: &integMetadata.Buckets.Target.ARN},
	}

	for _, bucket := range bucketCreates {
		*bucket.name = s3integ.GenerateBucketName()

		if err := s3integ.SetupBucket(s3Svc, *bucket.name); err != nil {
			return cleanup, err
		}

		// Compute ARN
		bARN := arn.ARN{
			Partition: "aws",
			Service:   "s3",
			Region:    s3Svc.SigningRegion,
			AccountID: integMetadata.AccountID,
			Resource:  fmt.Sprintf("bucket_name:%s", *bucket.name),
		}.String()

		*bucket.arn = bARN

		bucketName := *bucket.name
		cleanups = append(cleanups, func() {
			if err := s3integ.CleanupBucket(s3Svc, bucketName); err != nil {
				fmt.Fprintln(os.Stderr, err)
			}
		})
	}

	return cleanup, nil
}

func setupAccessPoints() (func(), error) {
	var cleanups []func()

	cleanup := func() {
		for i := range cleanups {
			cleanups[i]()
		}
	}

	creates := []struct {
		bucket string
		name   *string
		arn    *string
	}{
		{bucket: integMetadata.Buckets.Source.Name, name: &integMetadata.AccessPoints.Source.Name, arn: &integMetadata.AccessPoints.Source.ARN},
		{bucket: integMetadata.Buckets.Target.Name, name: &integMetadata.AccessPoints.Target.Name, arn: &integMetadata.AccessPoints.Target.ARN},
	}

	for _, ap := range creates {
		*ap.name = integration.UniqueID()

		err := s3integ.SetupAccessPoint(s3ControlSvc, integMetadata.AccountID, ap.bucket, *ap.name)
		if err != nil {
			return cleanup, err
		}

		// Compute ARN
		apARN := arn.ARN{
			Partition: "aws",
			Service:   "s3",
			Region:    s3ControlSvc.SigningRegion,
			AccountID: integMetadata.AccountID,
			Resource:  fmt.Sprintf("accesspoint/%s", *ap.name),
		}.String()

		*ap.arn = apARN

		apName := *ap.name
		cleanups = append(cleanups, func() {
			err := s3integ.CleanupAccessPoint(s3ControlSvc, integMetadata.AccountID, apName)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
			}
		})
	}

	return cleanup, nil
}

func putTestFile(t *testing.T, filename, key string, opts ...request.Option) {
	f, err := os.Open(filename)
	if err != nil {
		t.Fatalf("failed to open testfile, %v", err)
	}
	defer f.Close()

	putTestContent(t, f, key, opts...)
}

func putTestContent(t *testing.T, reader io.ReadSeeker, key string, opts ...request.Option) {
	t.Logf("uploading test file %s/%s", integMetadata.Buckets.Source.Name, key)
	_, err := s3Svc.PutObjectWithContext(context.Background(),
		&s3.PutObjectInput{
			Bucket: &integMetadata.Buckets.Source.Name,
			Key:    aws.String(key),
			Body:   reader,
		}, opts...)
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
}

func testWriteToObject(t *testing.T, bucket string, opts ...request.Option) {
	key := integration.UniqueID()

	_, err := s3Svc.PutObjectWithContext(context.Background(),
		&s3.PutObjectInput{
			Bucket: &bucket,
			Key:    &key,
			Body:   bytes.NewReader([]byte("hello world")),
		}, opts...)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	resp, err := s3Svc.GetObjectWithContext(context.Background(),
		&s3.GetObjectInput{
			Bucket: &bucket,
			Key:    &key,
		}, opts...)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	b, _ := ioutil.ReadAll(resp.Body)
	if e, a := []byte("hello world"), b; !bytes.Equal(e, a) {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func testPresignedGetPut(t *testing.T, bucket string, opts ...request.Option) {
	key := integration.UniqueID()

	putreq, _ := s3Svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket: &bucket,
		Key:    &key,
	})
	putreq.ApplyOptions(opts...)
	var err error

	// Presign a PUT request
	var puturl string
	puturl, err = putreq.Presign(5 * time.Minute)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	// PUT to the presigned URL with a body
	var puthttpreq *http.Request
	buf := bytes.NewReader([]byte("hello world"))
	puthttpreq, err = http.NewRequest("PUT", puturl, buf)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	var putresp *http.Response
	putresp, err = httpClient.Do(puthttpreq)
	if err != nil {
		t.Errorf("expect put with presign url no error, got %v", err)
	}
	if e, a := 200, putresp.StatusCode; e != a {
		t.Fatalf("expect %v, got %v", e, a)
	}

	// Presign a GET on the same URL
	getreq, _ := s3Svc.GetObjectRequest(&s3.GetObjectInput{
		Bucket: &bucket,
		Key:    &key,
	})
	getreq.ApplyOptions(opts...)

	var geturl string
	geturl, err = getreq.Presign(300 * time.Second)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	// Get the body
	var getresp *http.Response
	getresp, err = httpClient.Get(geturl)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	var b []byte
	defer getresp.Body.Close()
	b, err = ioutil.ReadAll(getresp.Body)
	if e, a := "hello world", string(b); e != a {
		t.Fatalf("expect %v, got %v", e, a)
	}
}

func testCopyObject(t *testing.T, sourceBucket string, targetBucket string, opts ...request.Option) {
	key := integration.UniqueID()

	_, err := s3Svc.PutObjectWithContext(context.Background(),
		&s3.PutObjectInput{
			Bucket: &sourceBucket,
			Key:    &key,
			Body:   bytes.NewReader([]byte("hello world")),
		}, opts...)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	_, err = s3Svc.CopyObjectWithContext(context.Background(),
		&s3.CopyObjectInput{
			Bucket:     &targetBucket,
			CopySource: aws.String("/" + sourceBucket + "/" + key),
			Key:        &key,
		}, opts...)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	resp, err := s3Svc.GetObjectWithContext(context.Background(),
		&s3.GetObjectInput{
			Bucket: &targetBucket,
			Key:    &key,
		}, opts...)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	b, _ := ioutil.ReadAll(resp.Body)
	if e, a := []byte("hello world"), b; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v, got %v", e, a)
	}
}
