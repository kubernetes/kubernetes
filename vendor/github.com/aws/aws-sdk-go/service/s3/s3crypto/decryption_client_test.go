package s3crypto_test

import (
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

func TestGetObjectGCM(t *testing.T) {
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	keyB64 := base64.StdEncoding.EncodeToString(key)
	// This is our KMS response
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, fmt.Sprintf("%s%s%s", `{"KeyId":"test-key-id","Plaintext":"`, keyB64, `"}`))
	}))
	defer ts.Close()

	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		Endpoint:         aws.String(ts.URL),
		DisableSSL:       aws.Bool(true),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})

	c := s3crypto.NewDecryptionClient(sess)
	if c == nil {
		t.Error("expected non-nil value")
	}
	input := &s3.GetObjectInput{
		Key:    aws.String("test"),
		Bucket: aws.String("test"),
	}
	req, out := c.GetObjectRequest(input)
	req.Handlers.Send.Clear()
	req.Handlers.Send.PushBack(func(r *request.Request) {
		iv, err := hex.DecodeString("0d18e06c7c725ac9e362e1ce")
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		b, err := hex.DecodeString("fa4362189661d163fcd6a56d8bf0405ad636ac1bbedd5cc3ee727dc2ab4a9489")
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Header: http.Header{
				http.CanonicalHeaderKey("x-amz-meta-x-amz-key-v2"):   []string{"SpFRES0JyU8BLZSKo51SrwILK4lhtZsWiMNjgO4WmoK+joMwZPG7Hw=="},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-iv"):       []string{base64.URLEncoding.EncodeToString(iv)},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-matdesc"):  []string{`{"kms_cmk_id":"arn:aws:kms:us-east-1:172259396726:key/a22a4b30-79f4-4b3d-bab4-a26d327a231b"}`},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-wrap-alg"): []string{s3crypto.KMSWrap},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-cek-alg"):  []string{s3crypto.AESGCMNoPadding},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-tag-len"):  []string{"128"},
			},
			Body: ioutil.NopCloser(bytes.NewBuffer(b)),
		}
		out.Metadata = make(map[string]*string)
		out.Metadata["x-amz-wrap-alg"] = aws.String(s3crypto.KMSWrap)
	})
	err := req.Send()
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	b, err := ioutil.ReadAll(out.Body)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	expected, err := hex.DecodeString("2db5168e932556f8089a0622981d017d")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if !bytes.Equal(expected, b) {
		t.Error("expected bytes to be equivalent")
	}
}

func TestGetObjectCBC(t *testing.T) {
	key, _ := hex.DecodeString("898be9cc5004ed0fa6e117c9a3099d31")
	keyB64 := base64.StdEncoding.EncodeToString(key)
	// This is our KMS response
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, fmt.Sprintf("%s%s%s", `{"KeyId":"test-key-id","Plaintext":"`, keyB64, `"}`))
	}))
	defer ts.Close()

	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		Endpoint:         aws.String(ts.URL),
		DisableSSL:       aws.Bool(true),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})

	c := s3crypto.NewDecryptionClient(sess)
	if c == nil {
		t.Error("expected non-nil value")
	}
	input := &s3.GetObjectInput{
		Key:    aws.String("test"),
		Bucket: aws.String("test"),
	}
	req, out := c.GetObjectRequest(input)
	req.Handlers.Send.Clear()
	req.Handlers.Send.PushBack(func(r *request.Request) {
		iv, err := hex.DecodeString("9dea7621945988f96491083849b068df")
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}
		b, err := hex.DecodeString("e232cd6ef50047801ee681ec30f61d53cfd6b0bca02fd03c1b234baa10ea82ac9dab8b960926433a19ce6dea08677e34")
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Header: http.Header{
				http.CanonicalHeaderKey("x-amz-meta-x-amz-key-v2"):   []string{"SpFRES0JyU8BLZSKo51SrwILK4lhtZsWiMNjgO4WmoK+joMwZPG7Hw=="},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-iv"):       []string{base64.URLEncoding.EncodeToString(iv)},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-matdesc"):  []string{`{"kms_cmk_id":"arn:aws:kms:us-east-1:172259396726:key/a22a4b30-79f4-4b3d-bab4-a26d327a231b"}`},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-wrap-alg"): []string{s3crypto.KMSWrap},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-cek-alg"):  []string{strings.Join([]string{s3crypto.AESCBC, s3crypto.AESCBCPadder.Name()}, "/")},
			},
			Body: ioutil.NopCloser(bytes.NewBuffer(b)),
		}
		out.Metadata = make(map[string]*string)
		out.Metadata["x-amz-wrap-alg"] = aws.String(s3crypto.KMSWrap)
	})
	err := req.Send()
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	b, err := ioutil.ReadAll(out.Body)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	expected, err := hex.DecodeString("0397f4f6820b1f9386f14403be5ac16e50213bd473b4874b9bcbf5f318ee686b1d")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if !bytes.Equal(expected, b) {
		t.Error("expected bytes to be equivalent")
	}
}

func TestGetObjectCBC2(t *testing.T) {
	key, _ := hex.DecodeString("8d70e92489c4e6cfb12261b4d17f4b85826da687fc8742fcf9f87fadb5b4cb89")
	keyB64 := base64.StdEncoding.EncodeToString(key)
	// This is our KMS response
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, fmt.Sprintf("%s%s%s", `{"KeyId":"test-key-id","Plaintext":"`, keyB64, `"}`))
	}))
	defer ts.Close()

	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		Endpoint:         aws.String(ts.URL),
		DisableSSL:       aws.Bool(true),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})

	c := s3crypto.NewDecryptionClient(sess)
	if c == nil {
		t.Error("expected non-nil value")
	}
	input := &s3.GetObjectInput{
		Key:    aws.String("test"),
		Bucket: aws.String("test"),
	}
	req, out := c.GetObjectRequest(input)
	req.Handlers.Send.Clear()
	req.Handlers.Send.PushBack(func(r *request.Request) {
		b, err := hex.DecodeString("fd0c71ecb7ed16a9bf42ea5f75501d416df608f190890c3b4d8897f24744cd7f9ea4a0b212e60634302450e1c5378f047ff753ccefe365d411c36339bf22e301fae4c3a6226719a4b93dc74c1af79d0296659b5d56c0892315f2c7cc30190220db1eaafae3920d6d9c65d0aa366499afc17af493454e141c6e0fbdeb6a990cb4")
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Header: http.Header{
				http.CanonicalHeaderKey("x-amz-meta-x-amz-key-v2"):   []string{"AQEDAHikdGvcj7Gil5VqAR/JWvvPp3ue26+t2vhWy4lL2hg4mAAAAH4wfAYJKoZIhvcNAQcGoG8wbQIBADBoBgkqhkiG9w0BBwEwHgYJYIZIAWUDBAEuMBEEDCcy43wCR0bSsnzTrAIBEIA7WdD2jxC3tCrK6TOdiEfbIN64m+UN7Velz4y0LRra5jn2U1CDClacwIpiBYuDp5ymPKO+ZqUGE0WEf20="},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-iv"):       []string{"EMMWJY8ZLcK/9FOj3iCpng=="},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-matdesc"):  []string{`{"kms_cmk_id":"arn:aws:kms:us-east-1:172259396726:key/a22a4b30-79f4-4b3d-bab4-a26d327a231b"}`},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-wrap-alg"): []string{s3crypto.KMSWrap},
				http.CanonicalHeaderKey("x-amz-meta-x-amz-cek-alg"):  []string{strings.Join([]string{s3crypto.AESCBC, s3crypto.AESCBCPadder.Name()}, "/")},
			},
			Body: ioutil.NopCloser(bytes.NewBuffer(b)),
		}
		fmt.Println("HEADER", r.HTTPResponse.Header)
		out.Metadata = make(map[string]*string)
		out.Metadata["x-amz-wrap-alg"] = aws.String(s3crypto.KMSWrap)
	})
	err := req.Send()
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	b, err := ioutil.ReadAll(out.Body)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	expected, err := hex.DecodeString("a6ccd3482f5ce25c9ddeb69437cd0acbc0bdda2ef8696d90781de2b35704543529871b2032e68ef1c5baed1769aba8d420d1aca181341b49b8b3587a6580cdf1d809c68f06735f7735c16691f4b70c967d68fc08195b81ad71bcc4df452fd0a5799c1e1234f92f1cd929fc072167ccf9f2ac85b93170932b32")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if !bytes.Equal(expected, b) {
		t.Error("expected bytes to be equivalent")
	}
}

func TestGetObjectWithContext(t *testing.T) {
	c := s3crypto.NewDecryptionClient(unit.Session)

	ctx := &awstesting.FakeContext{DoneCh: make(chan struct{})}
	ctx.Error = fmt.Errorf("context canceled")
	close(ctx.DoneCh)

	input := s3.GetObjectInput{
		Key:    aws.String("test"),
		Bucket: aws.String("test"),
	}
	_, err := c.GetObjectWithContext(ctx, &input)
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
