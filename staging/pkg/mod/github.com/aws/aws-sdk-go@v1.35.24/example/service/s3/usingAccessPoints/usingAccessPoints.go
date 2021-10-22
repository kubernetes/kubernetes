// +build example

package main

import (
	"fmt"
	"io/ioutil"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/arn"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3control"
)

const (
	bucketName  = "myBucketName"
	keyName     = "myKeyName"
	accountID   = "123456789012"
	accessPoint = "accesspointname"
)

func main() {
	sess := session.Must(session.NewSession())

	s3Svc := s3.New(sess)
	s3ControlSvc := s3control.New(sess)

	// Create an S3 Bucket
	fmt.Println("create s3 bucket")
	_, err := s3Svc.CreateBucket(&s3.CreateBucketInput{
		Bucket: aws.String(bucketName),
	})
	if err != nil {
		panic(fmt.Errorf("failed to create bucket: %v", err))
	}

	// Wait for S3 Bucket to Exist
	fmt.Println("wait for s3 bucket to exist")
	err = s3Svc.WaitUntilBucketExists(&s3.HeadBucketInput{
		Bucket: aws.String(bucketName),
	})
	if err != nil {
		panic(fmt.Sprintf("bucket failed to materialize: %v", err))
	}

	// Create an Access Point referring to the bucket
	fmt.Println("create an access point")
	_, err = s3ControlSvc.CreateAccessPoint(&s3control.CreateAccessPointInput{
		AccountId: aws.String(accountID),
		Bucket:    aws.String(bucketName),
		Name:      aws.String(accessPoint),
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create access point: %v", err))
	}

	// Use the SDK's ARN builder to create an ARN for the Access Point.
	apARN := arn.ARN{
		Partition: "aws",
		Service:   "s3",
		Region:    aws.StringValue(sess.Config.Region),
		AccountID: accountID,
		Resource:  "accesspoint/" + accessPoint,
	}

	// And Use Access Point ARN where bucket parameters are accepted
	fmt.Println("get object using access point")
	getObjectOutput, err := s3Svc.GetObject(&s3.GetObjectInput{
		Bucket: aws.String(apARN.String()),
		Key:    aws.String("somekey"),
	})
	if err != nil {
		panic(fmt.Sprintf("failed get object request: %v", err))
	}

	_, err = ioutil.ReadAll(getObjectOutput.Body)
	if err != nil {
		panic(fmt.Sprintf("failed to read object body: %v", err))
	}
}
