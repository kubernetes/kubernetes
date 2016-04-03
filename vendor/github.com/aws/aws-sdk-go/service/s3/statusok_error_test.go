package s3_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

const errMsg = `<Error><Code>ErrorCode</Code><Message>message body</Message><RequestId>requestID</RequestId><HostId>hostID=</HostId></Error>`

var lastModifiedTime = time.Date(2009, 11, 23, 0, 0, 0, 0, time.UTC)

func TestCopyObjectNoError(t *testing.T) {
	const successMsg = `
<?xml version="1.0" encoding="UTF-8"?>
<CopyObjectResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><LastModified>2009-11-23T0:00:00Z</LastModified><ETag>&quot;1da64c7f13d1e8dbeaea40b905fd586c&quot;</ETag></CopyObjectResult>`

	res, err := newCopyTestSvc(successMsg).CopyObject(&s3.CopyObjectInput{
		Bucket:     aws.String("bucketname"),
		CopySource: aws.String("bucketname/exists.txt"),
		Key:        aws.String("destination.txt"),
	})

	require.NoError(t, err)

	assert.Equal(t, fmt.Sprintf(`%q`, "1da64c7f13d1e8dbeaea40b905fd586c"), *res.CopyObjectResult.ETag)
	assert.Equal(t, lastModifiedTime, *res.CopyObjectResult.LastModified)
}

func TestCopyObjectError(t *testing.T) {
	_, err := newCopyTestSvc(errMsg).CopyObject(&s3.CopyObjectInput{
		Bucket:     aws.String("bucketname"),
		CopySource: aws.String("bucketname/doesnotexist.txt"),
		Key:        aws.String("destination.txt"),
	})

	require.Error(t, err)
	e := err.(awserr.Error)

	assert.Equal(t, "ErrorCode", e.Code())
	assert.Equal(t, "message body", e.Message())
}

func TestUploadPartCopySuccess(t *testing.T) {
	const successMsg = `
<?xml version="1.0" encoding="UTF-8"?>
<UploadPartCopyResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><LastModified>2009-11-23T0:00:00Z</LastModified><ETag>&quot;1da64c7f13d1e8dbeaea40b905fd586c&quot;</ETag></CopyObjectResult>`

	res, err := newCopyTestSvc(successMsg).UploadPartCopy(&s3.UploadPartCopyInput{
		Bucket:     aws.String("bucketname"),
		CopySource: aws.String("bucketname/doesnotexist.txt"),
		Key:        aws.String("destination.txt"),
		PartNumber: aws.Int64(0),
		UploadId:   aws.String("uploadID"),
	})

	require.NoError(t, err)

	assert.Equal(t, fmt.Sprintf(`%q`, "1da64c7f13d1e8dbeaea40b905fd586c"), *res.CopyPartResult.ETag)
	assert.Equal(t, lastModifiedTime, *res.CopyPartResult.LastModified)
}

func TestUploadPartCopyError(t *testing.T) {
	_, err := newCopyTestSvc(errMsg).UploadPartCopy(&s3.UploadPartCopyInput{
		Bucket:     aws.String("bucketname"),
		CopySource: aws.String("bucketname/doesnotexist.txt"),
		Key:        aws.String("destination.txt"),
		PartNumber: aws.Int64(0),
		UploadId:   aws.String("uploadID"),
	})

	require.Error(t, err)
	e := err.(awserr.Error)

	assert.Equal(t, "ErrorCode", e.Code())
	assert.Equal(t, "message body", e.Message())
}

func TestCompleteMultipartUploadSuccess(t *testing.T) {
	const successMsg = `
<?xml version="1.0" encoding="UTF-8"?>
<CompleteMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Location>locationName</Location><Bucket>bucketName</Bucket><Key>keyName</Key><ETag>"etagVal"</ETag></CompleteMultipartUploadResult>`
	res, err := newCopyTestSvc(successMsg).CompleteMultipartUpload(&s3.CompleteMultipartUploadInput{
		Bucket:   aws.String("bucketname"),
		Key:      aws.String("key"),
		UploadId: aws.String("uploadID"),
	})

	require.NoError(t, err)

	assert.Equal(t, `"etagVal"`, *res.ETag)
	assert.Equal(t, "bucketName", *res.Bucket)
	assert.Equal(t, "keyName", *res.Key)
	assert.Equal(t, "locationName", *res.Location)
}

func TestCompleteMultipartUploadError(t *testing.T) {
	_, err := newCopyTestSvc(errMsg).CompleteMultipartUpload(&s3.CompleteMultipartUploadInput{
		Bucket:   aws.String("bucketname"),
		Key:      aws.String("key"),
		UploadId: aws.String("uploadID"),
	})

	require.Error(t, err)
	e := err.(awserr.Error)

	assert.Equal(t, "ErrorCode", e.Code())
	assert.Equal(t, "message body", e.Message())
}

func newCopyTestSvc(errMsg string) *s3.S3 {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, errMsg, http.StatusOK)
	}))
	return s3.New(unit.Session, aws.NewConfig().
		WithEndpoint(server.URL).
		WithDisableSSL(true).
		WithMaxRetries(0).
		WithS3ForcePathStyle(true))
}
