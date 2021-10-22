// +build go1.7

package s3_test

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

const errMsg = `<Error><Code>ErrorCode</Code><Message>message body</Message><RequestId>requestID</RequestId><HostId>hostID=</HostId></Error>`
const xmlPreambleMsg = `<?xml version="1.0" encoding="UTF-8"?>`

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
		t.Fatalf("expected no error, but received %v", err)
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
	const statusCode = http.StatusOK

	svc := s3.New(unit.Session, &aws.Config{
		MaxRetries: aws.Int(0),
		SleepDelay: func(time.Duration) {},
	})

	svc.Handlers.Send.Swap(corehandlers.SendHandler.Name,
		request.NamedHandler{
			Name: "newCopyTestSvc",
			Fn: func(r *request.Request) {
				io.Copy(ioutil.Discard, r.HTTPRequest.Body)
				r.HTTPRequest.Body.Close()
				r.HTTPResponse = &http.Response{
					Status:     http.StatusText(statusCode),
					StatusCode: statusCode,
					Header:     http.Header{},
					Body:       ioutil.NopCloser(strings.NewReader(errMsg)),
				}
			},
		})

	return svc
}

func TestStatusOKPayloadHandling(t *testing.T) {
	cases := map[string]struct {
		Header   http.Header
		Payloads [][]byte
		OpCall   func(*s3.S3) error
		Err      string
	}{
		"200 error": {
			Header: http.Header{
				"Content-Length": []string{strconv.Itoa(len(errMsg))},
			},
			Payloads: [][]byte{[]byte(errMsg)},
			OpCall: func(c *s3.S3) error {
				_, err := c.CopyObject(&s3.CopyObjectInput{
					Bucket:     aws.String("bucketname"),
					CopySource: aws.String("bucketname/doesnotexist.txt"),
					Key:        aws.String("destination.txt"),
				})
				return err
			},
			Err: "ErrorCode: message body",
		},
		"200 error partial response": {
			Header: http.Header{
				"Content-Length": []string{strconv.Itoa(len(errMsg))},
			},
			Payloads: [][]byte{
				[]byte(errMsg[:20]),
			},
			OpCall: func(c *s3.S3) error {
				_, err := c.CopyObject(&s3.CopyObjectInput{
					Bucket:     aws.String("bucketname"),
					CopySource: aws.String("bucketname/doesnotexist.txt"),
					Key:        aws.String("destination.txt"),
				})
				return err
			},
			Err: "unexpected EOF",
		},
		"200 error multipart": {
			Header: http.Header{
				"Transfer-Encoding": []string{"chunked"},
			},
			Payloads: [][]byte{
				[]byte(errMsg[:20]),
				[]byte(errMsg[20:]),
			},
			OpCall: func(c *s3.S3) error {
				_, err := c.CopyObject(&s3.CopyObjectInput{
					Bucket:     aws.String("bucketname"),
					CopySource: aws.String("bucketname/doesnotexist.txt"),
					Key:        aws.String("destination.txt"),
				})
				return err
			},
			Err: "ErrorCode: message body",
		},
		"200 error multipart partial response": {
			Header: http.Header{
				"Transfer-Encoding": []string{"chunked"},
			},
			Payloads: [][]byte{
				[]byte(errMsg[:20]),
			},
			OpCall: func(c *s3.S3) error {
				_, err := c.CopyObject(&s3.CopyObjectInput{
					Bucket:     aws.String("bucketname"),
					CopySource: aws.String("bucketname/doesnotexist.txt"),
					Key:        aws.String("destination.txt"),
				})
				return err
			},
			Err: "XML syntax error",
		},
		"200 error multipart no payload": {
			Header: http.Header{
				"Transfer-Encoding": []string{"chunked"},
			},
			Payloads: [][]byte{},
			OpCall: func(c *s3.S3) error {
				_, err := c.CopyObject(&s3.CopyObjectInput{
					Bucket:     aws.String("bucketname"),
					CopySource: aws.String("bucketname/doesnotexist.txt"),
					Key:        aws.String("destination.txt"),
				})
				return err
			},
			Err: "empty response payload",
		},
		"response with only xml preamble": {
			Header: http.Header{
				"Transfer-Encoding": []string{"chunked"},
			},
			Payloads: [][]byte{
				[]byte(xmlPreambleMsg),
			},
			OpCall: func(c *s3.S3) error {
				_, err := c.CopyObject(&s3.CopyObjectInput{
					Bucket:     aws.String("bucketname"),
					CopySource: aws.String("bucketname/doesnotexist.txt"),
					Key:        aws.String("destination.txt"),
				})
				return err
			},
			Err: "empty response payload",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				ww := w.(interface {
					http.ResponseWriter
					http.Flusher
				})

				for k, vs := range c.Header {
					for _, v := range vs {
						ww.Header().Add(k, v)
					}
				}
				ww.WriteHeader(http.StatusOK)
				ww.Flush()

				for _, p := range c.Payloads {
					ww.Write(p)
					ww.Flush()
				}
			}))
			defer srv.Close()

			client := s3.New(unit.Session, &aws.Config{
				Endpoint:               &srv.URL,
				DisableSSL:             aws.Bool(true),
				DisableParamValidation: aws.Bool(true),
				S3ForcePathStyle:       aws.Bool(true),
			})

			err := c.OpCall(client)
			if len(c.Err) != 0 {
				if err == nil {
					t.Fatalf("expect error, got none")
				}
				if e, a := c.Err, err.Error(); !strings.Contains(a, e) {
					t.Fatalf("expect %v error in, %v", e, a)
				}
			} else if err != nil {
				t.Fatalf("expect no error, got %v", err)
			}
		})
	}
}
