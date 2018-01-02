package s3crypto_test

import (
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

func TestHeaderV2SaveStrategy(t *testing.T) {
	env := s3crypto.Envelope{
		CipherKey:             "Foo",
		IV:                    "Bar",
		MatDesc:               "{}",
		WrapAlg:               s3crypto.KMSWrap,
		CEKAlg:                s3crypto.AESGCMNoPadding,
		TagLen:                "128",
		UnencryptedMD5:        "hello",
		UnencryptedContentLen: "0",
	}
	params := &s3.PutObjectInput{}
	req := &request.Request{
		Params: params,
	}
	strat := s3crypto.HeaderV2SaveStrategy{}
	err := strat.Save(env, req)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	expected := map[string]*string{
		"X-Amz-Key-V2":                     aws.String("Foo"),
		"X-Amz-Iv":                         aws.String("Bar"),
		"X-Amz-Matdesc":                    aws.String("{}"),
		"X-Amz-Wrap-Alg":                   aws.String(s3crypto.KMSWrap),
		"X-Amz-Cek-Alg":                    aws.String(s3crypto.AESGCMNoPadding),
		"X-Amz-Tag-Len":                    aws.String("128"),
		"X-Amz-Unencrypted-Content-Md5":    aws.String("hello"),
		"X-Amz-Unencrypted-Content-Length": aws.String("0"),
	}

	if !reflect.DeepEqual(expected, params.Metadata) {
		t.Errorf("expected %v, but received %v", expected, params.Metadata)
	}
}
