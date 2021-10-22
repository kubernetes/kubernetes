package s3

import (
	"bytes"
	"crypto/md5"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/sdkio"
)

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) {
	return 0, fmt.Errorf("errorReader error")
}
func (errorReader) Seek(int64, int) (int64, error) {
	return 0, nil
}

func TestComputeBodyHases(t *testing.T) {
	bodyContent := []byte("bodyContent goes here")

	cases := []struct {
		Req               *request.Request
		ExpectMD5         string
		ExpectSHA256      string
		Error             string
		DisableContentMD5 bool
		Presigned         bool
	}{
		{
			Req: &request.Request{
				HTTPRequest: &http.Request{
					Header: http.Header{},
				},
				Body: bytes.NewReader(bodyContent),
			},
			ExpectMD5:    "CqD6NNPvoNOBT/5pkjtzOw==",
			ExpectSHA256: "3ff09c8b42a58a905e27835919ede45b61722e7cd400f30101bd9ed1a69a1825",
		},
		{
			Req: &request.Request{
				HTTPRequest: &http.Request{
					Header: func() http.Header {
						h := http.Header{}
						h.Set(contentMD5Header, "MD5AlreadySet")
						return h
					}(),
				},
				Body: bytes.NewReader(bodyContent),
			},
			ExpectMD5:    "MD5AlreadySet",
			ExpectSHA256: "3ff09c8b42a58a905e27835919ede45b61722e7cd400f30101bd9ed1a69a1825",
		},
		{
			Req: &request.Request{
				HTTPRequest: &http.Request{
					Header: func() http.Header {
						h := http.Header{}
						h.Set(contentSha256Header, "SHA256AlreadySet")
						return h
					}(),
				},
				Body: bytes.NewReader(bodyContent),
			},
			ExpectMD5:    "CqD6NNPvoNOBT/5pkjtzOw==",
			ExpectSHA256: "SHA256AlreadySet",
		},
		{
			Req: &request.Request{
				HTTPRequest: &http.Request{
					Header: func() http.Header {
						h := http.Header{}
						h.Set(contentMD5Header, "MD5AlreadySet")
						h.Set(contentSha256Header, "SHA256AlreadySet")
						return h
					}(),
				},
				Body: bytes.NewReader(bodyContent),
			},
			ExpectMD5:    "MD5AlreadySet",
			ExpectSHA256: "SHA256AlreadySet",
		},
		{
			Req: &request.Request{
				HTTPRequest: &http.Request{
					Header: http.Header{},
				},
				// Non-seekable reader
				Body: aws.ReadSeekCloser(bytes.NewBuffer(bodyContent)),
			},
			ExpectMD5:    "",
			ExpectSHA256: "",
		},
		{
			Req: &request.Request{
				HTTPRequest: &http.Request{
					Header: http.Header{},
				},
				// Empty seekable body
				Body: aws.ReadSeekCloser(bytes.NewReader(nil)),
			},
			ExpectMD5:    "1B2M2Y8AsgTpgAmY7PhCfg==",
			ExpectSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		},
		{
			Req: &request.Request{
				HTTPRequest: &http.Request{
					Header: http.Header{},
				},
				// failure while reading reader
				Body: errorReader{},
			},
			ExpectMD5:    "",
			ExpectSHA256: "",
			Error:        "errorReader error",
		},
		{
			// Disabled ContentMD5 validation
			Req: &request.Request{
				HTTPRequest: &http.Request{
					Header: http.Header{},
				},
				Body: bytes.NewReader(bodyContent),
			},
			ExpectMD5:         "",
			ExpectSHA256:      "",
			DisableContentMD5: true,
		},
		{
			// Disabled ContentMD5 validation
			Req: &request.Request{
				HTTPRequest: &http.Request{
					Header: http.Header{},
				},
				Body: bytes.NewReader(bodyContent),
			},
			ExpectMD5:    "",
			ExpectSHA256: "",
			Presigned:    true,
		},
	}

	for i, c := range cases {
		c.Req.Config.S3DisableContentMD5Validation = aws.Bool(c.DisableContentMD5)

		if c.Presigned {
			c.Req.ExpireTime = 10 * time.Minute
		}
		computeBodyHashes(c.Req)

		if e, a := c.ExpectMD5, c.Req.HTTPRequest.Header.Get(contentMD5Header); e != a {
			t.Errorf("%d, expect %v md5, got %v", i, e, a)
		}

		if e, a := c.ExpectSHA256, c.Req.HTTPRequest.Header.Get(contentSha256Header); e != a {
			t.Errorf("%d, expect %v sha256, got %v", i, e, a)
		}

		if len(c.Error) != 0 {
			if c.Req.Error == nil {
				t.Fatalf("%d, expect error, got none", i)
			}
			if e, a := c.Error, c.Req.Error.Error(); !strings.Contains(a, e) {
				t.Errorf("%d, expect %v error to be in %v", i, e, a)
			}

		} else if c.Req.Error != nil {
			t.Errorf("%d, expect no error, got %v", i, c.Req.Error)
		}
	}
}

func BenchmarkComputeBodyHashes(b *testing.B) {
	body := bytes.NewReader(make([]byte, 2*1024))
	req := &request.Request{
		HTTPRequest: &http.Request{
			Header: http.Header{},
		},
		Body: body,
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		computeBodyHashes(req)
		if req.Error != nil {
			b.Fatalf("expect no error, got %v", req.Error)
		}

		req.HTTPRequest.Header = http.Header{}
		body.Seek(0, sdkio.SeekStart)
	}
}

func TestAskForTxEncodingAppendMD5(t *testing.T) {
	cases := []struct {
		DisableContentMD5 bool
		Presigned         bool
	}{
		{DisableContentMD5: true},
		{DisableContentMD5: false},
		{Presigned: true},
	}

	for i, c := range cases {
		req := &request.Request{
			HTTPRequest: &http.Request{
				Header: http.Header{},
			},
			Config: aws.Config{
				S3DisableContentMD5Validation: aws.Bool(c.DisableContentMD5),
			},
		}
		if c.Presigned {
			req.ExpireTime = 10 * time.Minute
		}

		askForTxEncodingAppendMD5(req)

		v := req.HTTPRequest.Header.Get(amzTeHeader)

		expectHeader := !(c.DisableContentMD5 || c.Presigned)

		if e, a := expectHeader, len(v) != 0; e != a {
			t.Errorf("%d, expect %t disable content MD5, got %t, %s", i, e, a, v)
		}
	}
}

func TestUseMD5ValidationReader(t *testing.T) {
	body := []byte("create a really cool md5 checksum of me")
	bodySum := md5.Sum(body)
	bodyWithSum := append(body, bodySum[:]...)

	emptyBodySum := md5.Sum([]byte{})

	cases := []struct {
		Req      *request.Request
		Error    string
		Validate func(outupt interface{}) error
	}{
		{
			// Positive: Use Validation reader
			Req: &request.Request{
				HTTPResponse: &http.Response{
					Header: func() http.Header {
						h := http.Header{}
						h.Set(amzTxEncodingHeader, appendMD5TxEncoding)
						return h
					}(),
				},
				Data: &GetObjectOutput{
					Body:          ioutil.NopCloser(bytes.NewReader(bodyWithSum)),
					ContentLength: aws.Int64(int64(len(bodyWithSum))),
				},
			},
			Validate: func(output interface{}) error {
				getObjOut := output.(*GetObjectOutput)
				reader, ok := getObjOut.Body.(*md5ValidationReader)
				if !ok {
					return fmt.Errorf("expect %T updated body reader, got %T",
						(*md5ValidationReader)(nil), getObjOut.Body)
				}

				if reader.rawReader == nil {
					return fmt.Errorf("expect rawReader not to be nil")
				}
				if reader.payload == nil {
					return fmt.Errorf("expect payload not to be nil")
				}
				if e, a := int64(len(bodyWithSum)-md5.Size), reader.payloadLen; e != a {
					return fmt.Errorf("expect %v payload len, got %v", e, a)
				}
				if reader.hash == nil {
					return fmt.Errorf("expect hash not to be nil")
				}

				return nil
			},
		},
		{
			// Positive: Use Validation reader, empty object
			Req: &request.Request{
				HTTPResponse: &http.Response{
					Header: func() http.Header {
						h := http.Header{}
						h.Set(amzTxEncodingHeader, appendMD5TxEncoding)
						return h
					}(),
				},
				Data: &GetObjectOutput{
					Body:          ioutil.NopCloser(bytes.NewReader(emptyBodySum[:])),
					ContentLength: aws.Int64(int64(len(emptyBodySum[:]))),
				},
			},
			Validate: func(output interface{}) error {
				getObjOut := output.(*GetObjectOutput)
				reader, ok := getObjOut.Body.(*md5ValidationReader)
				if !ok {
					return fmt.Errorf("expect %T updated body reader, got %T",
						(*md5ValidationReader)(nil), getObjOut.Body)
				}

				if reader.rawReader == nil {
					return fmt.Errorf("expect rawReader not to be nil")
				}
				if reader.payload == nil {
					return fmt.Errorf("expect payload not to be nil")
				}
				if e, a := int64(len(emptyBodySum)-md5.Size), reader.payloadLen; e != a {
					return fmt.Errorf("expect %v payload len, got %v", e, a)
				}
				if reader.hash == nil {
					return fmt.Errorf("expect hash not to be nil")
				}

				return nil
			},
		},
		{
			// Negative: amzTxEncoding header not set
			Req: &request.Request{
				HTTPResponse: &http.Response{
					Header: http.Header{},
				},
				Data: &GetObjectOutput{
					Body:          ioutil.NopCloser(bytes.NewReader(body)),
					ContentLength: aws.Int64(int64(len(body))),
				},
			},
			Validate: func(output interface{}) error {
				getObjOut := output.(*GetObjectOutput)
				reader, ok := getObjOut.Body.(*md5ValidationReader)
				if ok {
					return fmt.Errorf("expect body reader not to be %T",
						reader)
				}

				return nil
			},
		},
		{
			// Negative: Not GetObjectOutput type.
			Req: &request.Request{
				Operation: &request.Operation{
					Name: "PutObject",
				},
				HTTPResponse: &http.Response{
					Header: func() http.Header {
						h := http.Header{}
						h.Set(amzTxEncodingHeader, appendMD5TxEncoding)
						return h
					}(),
				},
				Data: &PutObjectOutput{},
			},
			Error: "header received on unsupported API",
			Validate: func(output interface{}) error {
				_, ok := output.(*PutObjectOutput)
				if !ok {
					return fmt.Errorf("expect %T output not to change, got %T",
						(*PutObjectOutput)(nil), output)
				}

				return nil
			},
		},
		{
			// Negative: invalid content length.
			Req: &request.Request{
				HTTPResponse: &http.Response{
					Header: func() http.Header {
						h := http.Header{}
						h.Set(amzTxEncodingHeader, appendMD5TxEncoding)
						return h
					}(),
				},
				Data: &GetObjectOutput{
					Body:          ioutil.NopCloser(bytes.NewReader(bodyWithSum)),
					ContentLength: aws.Int64(-1),
				},
			},
			Error: "invalid Content-Length -1",
			Validate: func(output interface{}) error {
				getObjOut := output.(*GetObjectOutput)
				reader, ok := getObjOut.Body.(*md5ValidationReader)
				if ok {
					return fmt.Errorf("expect body reader not to be %T",
						reader)
				}
				return nil
			},
		},
		{
			// Negative: invalid content length, < md5.Size.
			Req: &request.Request{
				HTTPResponse: &http.Response{
					Header: func() http.Header {
						h := http.Header{}
						h.Set(amzTxEncodingHeader, appendMD5TxEncoding)
						return h
					}(),
				},
				Data: &GetObjectOutput{
					Body:          ioutil.NopCloser(bytes.NewReader(make([]byte, 5))),
					ContentLength: aws.Int64(5),
				},
			},
			Error: "invalid Content-Length 5",
			Validate: func(output interface{}) error {
				getObjOut := output.(*GetObjectOutput)
				reader, ok := getObjOut.Body.(*md5ValidationReader)
				if ok {
					return fmt.Errorf("expect body reader not to be %T",
						reader)
				}
				return nil
			},
		},
	}

	for i, c := range cases {
		useMD5ValidationReader(c.Req)
		if len(c.Error) != 0 {
			if c.Req.Error == nil {
				t.Fatalf("%d, expect error, got none", i)
			}
			if e, a := c.Error, c.Req.Error.Error(); !strings.Contains(a, e) {
				t.Errorf("%d, expect %v error to be in %v", i, e, a)
			}
		} else if c.Req.Error != nil {
			t.Errorf("%d, expect no error, got %v", i, c.Req.Error)
		}

		if c.Validate != nil {
			if err := c.Validate(c.Req.Data); err != nil {
				t.Errorf("%d, expect Data to validate, got %v", i, err)
			}
		}
	}
}

func TestReaderMD5Validation(t *testing.T) {
	body := []byte("create a really cool md5 checksum of me")
	bodySum := md5.Sum(body)
	bodyWithSum := append(body, bodySum[:]...)
	emptyBodySum := md5.Sum([]byte{})
	badBodySum := append(body, emptyBodySum[:]...)

	cases := []struct {
		Content       []byte
		ContentReader io.ReadCloser
		PayloadLen    int64
		Error         string
	}{
		{
			Content:    bodyWithSum,
			PayloadLen: int64(len(body)),
		},
		{
			Content:    emptyBodySum[:],
			PayloadLen: 0,
		},
		{
			Content:    badBodySum,
			PayloadLen: int64(len(body)),
			Error:      "expected MD5 checksum",
		},
		{
			Content:    emptyBodySum[:len(emptyBodySum)-2],
			PayloadLen: 0,
			Error:      "unexpected EOF",
		},
		{
			Content:    body,
			PayloadLen: int64(len(body) * 2),
			Error:      "unexpected EOF",
		},
		{
			ContentReader: ioutil.NopCloser(errorReader{}),
			PayloadLen:    int64(len(body)),
			Error:         "errorReader error",
		},
	}

	for i, c := range cases {
		reader := c.ContentReader
		if reader == nil {
			reader = ioutil.NopCloser(bytes.NewReader(c.Content))
		}
		v := newMD5ValidationReader(reader, c.PayloadLen)

		var actual bytes.Buffer
		n, err := io.Copy(&actual, v)
		if len(c.Error) != 0 {
			if err == nil {
				t.Errorf("%d, expect error, got none", i)
			}
			if e, a := c.Error, err.Error(); !strings.Contains(a, e) {
				t.Errorf("%d, expect %v error to be in %v", i, e, a)
			}
			continue
		} else if err != nil {
			t.Errorf("%d, expect no error, got %v", i, err)
			continue
		}
		if e, a := c.PayloadLen, n; e != a {
			t.Errorf("%d, expect %v len, got %v", i, e, a)
		}

		if e, a := c.Content[:c.PayloadLen], actual.Bytes(); !bytes.Equal(e, a) {
			t.Errorf("%d, expect:\n%v\nactual:\n%v", i, e, a)
		}
	}
}
