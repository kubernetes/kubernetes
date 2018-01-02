package s3crypto_test

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/kms"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

func TestDefaultConfigValues(t *testing.T) {
	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})
	svc := kms.New(sess)
	handler := s3crypto.NewKMSKeyGenerator(svc, "testid")

	c := s3crypto.NewEncryptionClient(sess, s3crypto.AESGCMContentCipherBuilder(handler))

	if c == nil {
		t.Error("expected non-vil client value")
	}
	if c.ContentCipherBuilder == nil {
		t.Error("expected non-vil content cipher builder value")
	}
	if c.SaveStrategy == nil {
		t.Error("expected non-vil save strategy value")
	}
}

func TestPutObject(t *testing.T) {
	size := 1024 * 1024
	data := make([]byte, size)
	expected := bytes.Repeat([]byte{1}, size)
	generator := mockGenerator{}
	cb := mockCipherBuilder{generator}
	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})
	c := s3crypto.NewEncryptionClient(sess, cb)
	if c == nil {
		t.Error("expected non-vil client value")
	}
	input := &s3.PutObjectInput{
		Key:    aws.String("test"),
		Bucket: aws.String("test"),
		Body:   bytes.NewReader(data),
	}
	req, _ := c.PutObjectRequest(input)
	req.Handlers.Send.Clear()
	req.Handlers.Send.PushBack(func(r *request.Request) {
		r.Error = errors.New("stop")
		r.HTTPResponse = &http.Response{
			StatusCode: 200,
		}
	})
	err := req.Send()
	if e, a := "stop", err.Error(); e != a {
		t.Errorf("expected %s error, but received %s", e, a)
	}
	b, err := ioutil.ReadAll(req.HTTPRequest.Body)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if !bytes.Equal(expected, b) {
		t.Error("expected bytes to be equivalent, but received otherwise")
	}
}

func TestPutObjectWithContext(t *testing.T) {
	generator := mockGenerator{}
	cb := mockCipherBuilder{generator}

	c := s3crypto.NewEncryptionClient(unit.Session, cb)

	ctx := &awstesting.FakeContext{DoneCh: make(chan struct{})}
	ctx.Error = fmt.Errorf("context canceled")
	close(ctx.DoneCh)

	input := s3.PutObjectInput{
		Bucket: aws.String("test"),
		Key:    aws.String("test"),
		Body:   bytes.NewReader([]byte{}),
	}
	_, err := c.PutObjectWithContext(ctx, &input)
	if err == nil {
		t.Fatalf("expected error, did not get one")
	}
	aerr := err.(awserr.Error)
	if e, a := request.CanceledErrorCode, aerr.Code(); e != a {
		t.Errorf("expected error code %q, got %q", e, a)
	}
	if e, a := "canceled", aerr.Message(); !strings.Contains(a, e) {
		t.Errorf("expected error message to contain %q, but did not %q", e, a)
	}
}
