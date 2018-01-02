package s3_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

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

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if e, a := fmt.Sprintf(`%q`, "1da64c7f13d1e8dbeaea40b905fd586c"), *res.CopyObjectResult.ETag; e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := lastModifiedTime, *res.CopyObjectResult.LastModified; !e.Equal(a) {
		t.Errorf("expected %v, but received %v", e, a)
	}
}

func TestCopyObjectError(t *testing.T) {
	_, err := newCopyTestSvc(errMsg).CopyObject(&s3.CopyObjectInput{
		Bucket:     aws.String("bucketname"),
		CopySource: aws.String("bucketname/doesnotexist.txt"),
		Key:        aws.String("destination.txt"),
	})

	if err == nil {
		t.Error("expected error, but received none")
	}
	e := err.(awserr.Error)

	if e, a := "ErrorCode", e.Code(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "message body", e.Message(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
}

func TestUploadPartCopySuccess(t *testing.T) {
	const successMsg = `
<?xml version="1.0" encoding="UTF-8"?>
<UploadPartCopyResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><LastModified>2009-11-23T0:00:00Z</LastModified><ETag>&quot;1da64c7f13d1e8dbeaea40b905fd586c&quot;</ETag></UploadPartCopyResult>`

	res, err := newCopyTestSvc(successMsg).UploadPartCopy(&s3.UploadPartCopyInput{
		Bucket:     aws.String("bucketname"),
		CopySource: aws.String("bucketname/doesnotexist.txt"),
		Key:        aws.String("destination.txt"),
		PartNumber: aws.Int64(0),
		UploadId:   aws.String("uploadID"),
	})

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := fmt.Sprintf(`%q`, "1da64c7f13d1e8dbeaea40b905fd586c"), *res.CopyPartResult.ETag; e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := lastModifiedTime, *res.CopyPartResult.LastModified; !e.Equal(a) {
		t.Errorf("expected %v, but received %v", e, a)
	}
}

func TestUploadPartCopyError(t *testing.T) {
	_, err := newCopyTestSvc(errMsg).UploadPartCopy(&s3.UploadPartCopyInput{
		Bucket:     aws.String("bucketname"),
		CopySource: aws.String("bucketname/doesnotexist.txt"),
		Key:        aws.String("destination.txt"),
		PartNumber: aws.Int64(0),
		UploadId:   aws.String("uploadID"),
	})

	if err == nil {
		t.Error("expected an error")
	}
	e := err.(awserr.Error)

	if e, a := "ErrorCode", e.Code(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "message body", e.Message(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
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

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := `"etagVal"`, *res.ETag; e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "bucketName", *res.Bucket; e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "keyName", *res.Key; e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "locationName", *res.Location; e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
}

func TestCompleteMultipartUploadError(t *testing.T) {
	_, err := newCopyTestSvc(errMsg).CompleteMultipartUpload(&s3.CompleteMultipartUploadInput{
		Bucket:   aws.String("bucketname"),
		Key:      aws.String("key"),
		UploadId: aws.String("uploadID"),
	})

	if err == nil {
		t.Error("expected an error")
	}
	e := err.(awserr.Error)

	if e, a := "ErrorCode", e.Code(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "message body", e.Message(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
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
