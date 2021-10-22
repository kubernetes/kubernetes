// +build integration

package s3_test

import (
	"bytes"
	"crypto/md5"
	"encoding/base64"
	"fmt"
	"io"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/s3"
)

func base64Sum(content []byte) string {
	sum := md5.Sum(content)
	return base64.StdEncoding.EncodeToString(sum[:])
}

func SkipTestContentMD5Validate(t *testing.T) {
	body := []byte("really cool body content")

	cases := []struct {
		Name     string
		Body     []byte
		Sum64    string
		RangeGet []int64
	}{
		{
			Body:  body,
			Sum64: base64Sum(body),
			Name:  "contentMD5validation.pop",
		},
		{
			Body:  []byte{},
			Sum64: base64Sum([]byte{}),
			Name:  "contentMD5validation.empty",
		},
		{
			Body:     body,
			Sum64:    base64Sum(body),
			RangeGet: []int64{0, 9},
			Name:     "contentMD5validation.range",
		},
	}

	for i, c := range cases {
		keyName := aws.String(c.Name)
		req, _ := s3Svc.PutObjectRequest(&s3.PutObjectInput{
			Bucket: &integMetadata.Buckets.Source.Name,
			Key:    keyName,
			Body:   bytes.NewReader(c.Body),
		})

		req.Build()
		if e, a := c.Sum64, req.HTTPRequest.Header.Get("Content-Md5"); e != a {
			t.Errorf("%d, expect %v sum, got %v", i, e, a)
		}

		if err := req.Send(); err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}

		getObjIn := &s3.GetObjectInput{
			Bucket: &integMetadata.Buckets.Source.Name,
			Key:    keyName,
		}

		expectBody := c.Body
		if c.RangeGet != nil {
			getObjIn.Range = aws.String(fmt.Sprintf("bytes=%d-%d", c.RangeGet[0], c.RangeGet[1]-1))
			expectBody = c.Body[c.RangeGet[0]:c.RangeGet[1]]
		}

		getReq, getOut := s3Svc.GetObjectRequest(getObjIn)

		getReq.Build()
		if e, a := "append-md5", getReq.HTTPRequest.Header.Get("X-Amz-Te"); e != a {
			t.Errorf("%d, expect %v encoding, got %v", i, e, a)
		}
		if err := getReq.Send(); err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}
		defer getOut.Body.Close()

		if e, a := "append-md5", getReq.HTTPResponse.Header.Get("X-Amz-Transfer-Encoding"); e != a {
			t.Fatalf("%d, expect response tx encoding header %v, got %v", i, e, a)
		}

		var readBody bytes.Buffer
		_, err := io.Copy(&readBody, getOut.Body)
		if err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}

		if e, a := expectBody, readBody.Bytes(); !bytes.Equal(e, a) {
			t.Errorf("%d, expect %v body, got %v", i, e, a)
		}
	}
}
