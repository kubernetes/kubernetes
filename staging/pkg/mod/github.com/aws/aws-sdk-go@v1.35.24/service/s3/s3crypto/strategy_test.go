package s3crypto_test

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

func TestHeaderV2SaveStrategy(t *testing.T) {
	cases := []struct {
		env      s3crypto.Envelope
		expected map[string]*string
	}{
		{
			s3crypto.Envelope{
				CipherKey:             "Foo",
				IV:                    "Bar",
				MatDesc:               "{}",
				WrapAlg:               s3crypto.KMSWrap,
				CEKAlg:                s3crypto.AESGCMNoPadding,
				TagLen:                "128",
				UnencryptedMD5:        "hello",
				UnencryptedContentLen: "0",
			},
			map[string]*string{
				"X-Amz-Key-V2":                     aws.String("Foo"),
				"X-Amz-Iv":                         aws.String("Bar"),
				"X-Amz-Matdesc":                    aws.String("{}"),
				"X-Amz-Wrap-Alg":                   aws.String(s3crypto.KMSWrap),
				"X-Amz-Cek-Alg":                    aws.String(s3crypto.AESGCMNoPadding),
				"X-Amz-Tag-Len":                    aws.String("128"),
				"X-Amz-Unencrypted-Content-Length": aws.String("0"),
			},
		},
		{
			s3crypto.Envelope{
				CipherKey:             "Foo",
				IV:                    "Bar",
				MatDesc:               "{}",
				WrapAlg:               s3crypto.KMSWrap,
				CEKAlg:                s3crypto.AESGCMNoPadding,
				UnencryptedMD5:        "hello",
				UnencryptedContentLen: "0",
			},
			map[string]*string{
				"X-Amz-Key-V2":                     aws.String("Foo"),
				"X-Amz-Iv":                         aws.String("Bar"),
				"X-Amz-Matdesc":                    aws.String("{}"),
				"X-Amz-Wrap-Alg":                   aws.String(s3crypto.KMSWrap),
				"X-Amz-Cek-Alg":                    aws.String(s3crypto.AESGCMNoPadding),
				"X-Amz-Unencrypted-Content-Length": aws.String("0"),
			},
		},
	}

	for _, c := range cases {
		params := &s3.PutObjectInput{}
		req := &request.Request{
			Params: params,
		}
		strat := s3crypto.HeaderV2SaveStrategy{}
		err := strat.Save(c.env, req)
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		if !reflect.DeepEqual(c.expected, params.Metadata) {
			t.Errorf("expected %v, but received %v", c.expected, params.Metadata)
		}
	}
}

func TestS3SaveStrategy(t *testing.T) {
	cases := []struct {
		env      s3crypto.Envelope
		expected s3crypto.Envelope
	}{
		{
			s3crypto.Envelope{
				CipherKey:             "Foo",
				IV:                    "Bar",
				MatDesc:               "{}",
				WrapAlg:               s3crypto.KMSWrap,
				CEKAlg:                s3crypto.AESGCMNoPadding,
				TagLen:                "128",
				UnencryptedMD5:        "hello",
				UnencryptedContentLen: "0",
			},
			s3crypto.Envelope{
				CipherKey:             "Foo",
				IV:                    "Bar",
				MatDesc:               "{}",
				WrapAlg:               s3crypto.KMSWrap,
				CEKAlg:                s3crypto.AESGCMNoPadding,
				TagLen:                "128",
				UnencryptedContentLen: "0",
			},
		},
		{
			s3crypto.Envelope{
				CipherKey:             "Foo",
				IV:                    "Bar",
				MatDesc:               "{}",
				WrapAlg:               s3crypto.KMSWrap,
				CEKAlg:                s3crypto.AESGCMNoPadding,
				UnencryptedMD5:        "hello",
				UnencryptedContentLen: "0",
			},
			s3crypto.Envelope{
				CipherKey:             "Foo",
				IV:                    "Bar",
				MatDesc:               "{}",
				WrapAlg:               s3crypto.KMSWrap,
				CEKAlg:                s3crypto.AESGCMNoPadding,
				UnencryptedContentLen: "0",
			},
		},
	}

	for _, c := range cases {
		params := &s3.PutObjectInput{
			Bucket: aws.String("fooBucket"),
			Key:    aws.String("barKey"),
		}
		req := &request.Request{
			Params: params,
		}

		client := s3.New(unit.Session)

		client.Handlers.Send.Clear()
		client.Handlers.Unmarshal.Clear()
		client.Handlers.UnmarshalMeta.Clear()
		client.Handlers.UnmarshalError.Clear()
		client.Handlers.Send.PushBack(func(r *request.Request) {
			bodyBytes, err := ioutil.ReadAll(r.Body)
			if err != nil {
				r.HTTPResponse = &http.Response{
					StatusCode: 500,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte(err.Error()))),
				}
				return
			}

			var actual s3crypto.Envelope
			err = json.Unmarshal(bodyBytes, &actual)
			if err != nil {
				r.HTTPResponse = &http.Response{
					StatusCode: 500,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte(err.Error()))),
				}
				return
			}

			if e, a := c.expected, actual; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, got %v", e, a)
			}

			r.HTTPResponse = &http.Response{
				StatusCode: 200,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}
		})

		strat := s3crypto.S3SaveStrategy{Client: client}
		err := strat.Save(c.env, req)
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}
	}
}
