// +build example

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

// server.go is an example of a service that vends lists for requests for presigned
// URLs for S3 objects. The service supports two S3 operations, "GetObject" and
// "PutObject".
//
// Example GetObject request to the service for the object with the key "MyObjectKey":
//
//   curl -v "http://127.0.0.1:8080/presign/my-object/key?method=GET"
//
// Example PutObject request to the service for the object with the key "MyObjectKey":
//
//   curl -v "http://127.0.0.1:8080/presign/my-object/key?method=PUT&contentLength=1024"
//
// Use "--help" command line argument flag to see all options and defaults.
//
// Usage:
//   go run -tags example service.go -b myBucket
func main() {
	addr, bucket, region := loadConfig()

	// Create a AWS SDK for Go Session that will load credentials using the SDK's
	// default credential change.
	sess := session.Must(session.NewSession())

	// Use the GetBucketRegion utility to lookup the bucket's region automatically.
	// The service.go will only do this correctly for AWS regions. For AWS China
	// and AWS Gov Cloud the region needs to be specified to let the service know
	// to look in those partitions instead of AWS.
	if len(region) == 0 {
		var err error
		region, err = s3manager.GetBucketRegion(aws.BackgroundContext(), sess, bucket, endpoints.UsWest2RegionID)
		if err != nil {
			exitError(fmt.Errorf("failed to get bucket region, %v", err))
		}
	}

	// Create a new S3 service client that will be use by the service to generate
	// presigned URLs with. Not actual API requests will be made with this client.
	// The credentials loaded when the Session was created above will be used
	// to sign the requests with.
	s3Svc := s3.New(sess, &aws.Config{
		Region: aws.String(region),
	})

	// Start the server listening and serve presigned URLs for GetObject and
	// PutObject requests.
	if err := listenAndServe(addr, bucket, s3Svc); err != nil {
		exitError(err)
	}
}

func loadConfig() (addr, bucket, region string) {
	flag.StringVar(&bucket, "b", "", "S3 `bucket` object should be uploaded to.")
	flag.StringVar(&region, "r", "", "AWS `region` the bucket exists in, If not set region will be looked up, only valid for AWS Regions, not AWS China or Gov Cloud.")
	flag.StringVar(&addr, "a", "127.0.0.1:8080", "The TCP `address` the server will be started on.")
	flag.Parse()

	if len(bucket) == 0 {
		fmt.Fprintln(os.Stderr, "bucket is required")
		flag.PrintDefaults()
		os.Exit(1)
	}

	return addr, bucket, region
}

func listenAndServe(addr, bucket string, svc s3iface.S3API) error {
	l, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to start service listener, %v", err)
	}

	const presignPath = "/presign/"

	// Create the HTTP handler for the "/presign/" path prefix. This will handle
	// all requests on this path, extracting the object's key from the path.
	http.HandleFunc(presignPath, func(w http.ResponseWriter, r *http.Request) {
		var u string
		var err error
		var signedHeaders http.Header

		query := r.URL.Query()

		var contentLen int64
		// Optionally the Content-Length header can be included with the signature
		// of the request. This is helpful to ensure the content uploaded is the
		// size that is expected. Constraints like these can be further expanded
		// with headers such as `Content-Type`. These can be enforced by the service
		// requiring the client to satisfying those constraints when uploading
		//
		// In addition the client could provide the service with a SHA256 of the
		// content to be uploaded. This prevents any other third party from uploading
		// anything else with the presigned URL
		if contLenStr := query.Get("contentLength"); len(contLenStr) > 0 {
			contentLen, err = strconv.ParseInt(contLenStr, 10, 64)
			if err != nil {
				fmt.Fprintf(os.Stderr, "unable to parse request content length, %v", err)
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}

		// Extract the object key from the path
		key := strings.Replace(r.URL.Path, presignPath, "", 1)
		method := query.Get("method")

		switch method {
		case "PUT":
			// For creating PutObject presigned URLs
			fmt.Println("Received request to presign PutObject for,", key)
			sdkReq, _ := svc.PutObjectRequest(&s3.PutObjectInput{
				Bucket: aws.String(bucket),
				Key:    aws.String(key),

				// If ContentLength is 0 the header will not be included in the signature.
				ContentLength: aws.Int64(contentLen),
			})
			u, signedHeaders, err = sdkReq.PresignRequest(15 * time.Minute)
		case "GET":
			// For creating GetObject presigned URLs
			fmt.Println("Received request to presign GetObject for,", key)
			sdkReq, _ := svc.GetObjectRequest(&s3.GetObjectInput{
				Bucket: aws.String(bucket),
				Key:    aws.String(key),
			})
			u, signedHeaders, err = sdkReq.PresignRequest(15 * time.Minute)
		default:
			fmt.Fprintf(os.Stderr, "invalid method provided, %s, %v\n", method, err)
			err = fmt.Errorf("invalid request")
		}

		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Create the response back to the client with the information on the
		// presigned request and additional headers to include.
		if err := json.NewEncoder(w).Encode(PresignResp{
			Method: method,
			URL:    u,
			Header: signedHeaders,
		}); err != nil {
			fmt.Fprintf(os.Stderr, "failed to encode presign response, %v", err)
		}
	})

	fmt.Println("Starting Server On:", "http://"+l.Addr().String())

	s := &http.Server{}
	return s.Serve(l)
}

// PresignResp provides the Go representation of the JSON value that will be
// sent to the client.
type PresignResp struct {
	Method, URL string
	Header      http.Header
}

func exitError(err error) {
	fmt.Fprintln(os.Stderr, err.Error())
	os.Exit(1)
}
