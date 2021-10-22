// +build go1.7

package s3crypto

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/kms"
	"github.com/aws/aws-sdk-go/service/s3"
)

func sessionWithLogCheck(message string) (*session.Session, *bool) {
	gotWarning := false

	u := unit.Session.Copy(&aws.Config{Logger: aws.LoggerFunc(func(i ...interface{}) {
		if len(i) == 0 {
			return
		}
		s, ok := i[0].(string)
		if !ok {
			return
		}
		if s == message {
			gotWarning = true
		}
	})})

	return u, &gotWarning
}

func TestNewEncryptionClientV2(t *testing.T) {
	tUnit, gotWarning := sessionWithLogCheck(customTypeWarningMessage)

	mcb := AESGCMContentCipherBuilderV2(NewKMSContextKeyGenerator(nil, "id", nil))
	v2, err := NewEncryptionClientV2(tUnit, mcb)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if v2 == nil {
		t.Fatal("expected client to not be nil")
	}

	if *gotWarning {
		t.Errorf("expected no warning for aws provided custom cipher builder")
	}

	if !reflect.DeepEqual(mcb, v2.options.ContentCipherBuilder) {
		t.Errorf("content cipher builder did not match provided value")
	}

	_, ok := v2.options.SaveStrategy.(HeaderV2SaveStrategy)
	if !ok {
		t.Errorf("expected default save strategy to be s3 header strategy")
	}

	if v2.options.S3Client == nil {
		t.Errorf("expected s3 client not be nil")
	}

	if e, a := DefaultMinFileSize, v2.options.MinFileSize; int64(e) != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	if e, a := "", v2.options.TempFolderPath; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestNewEncryptionClientV2_NonDefaults(t *testing.T) {
	tUnit, gotWarning := sessionWithLogCheck(customTypeWarningMessage)

	s3Client := s3.New(tUnit)

	mcb := mockCipherBuilderV2{}
	v2, err := NewEncryptionClientV2(tUnit, nil, func(clientOptions *EncryptionClientOptions) {
		clientOptions.S3Client = s3Client
		clientOptions.ContentCipherBuilder = mcb
		clientOptions.TempFolderPath = "/mock/path"
		clientOptions.MinFileSize = 42
		clientOptions.SaveStrategy = S3SaveStrategy{}
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if v2 == nil {
		t.Fatal("expected client to not be nil")
	}

	if !*gotWarning {
		t.Errorf("expected warning for custom provided content cipher builder")
	}

	if !reflect.DeepEqual(mcb, v2.options.ContentCipherBuilder) {
		t.Errorf("content cipher builder did not match provided value")
	}

	_, ok := v2.options.SaveStrategy.(S3SaveStrategy)
	if !ok {
		t.Errorf("expected default save strategy to be s3 header strategy")
	}

	if v2.options.S3Client != s3Client {
		t.Errorf("expected s3 client not be nil")
	}

	if e, a := 42, v2.options.MinFileSize; int64(e) != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	if e, a := "/mock/path", v2.options.TempFolderPath; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

// cdgWithStaticTestIV is a test structure that wraps a CipherDataGeneratorWithCEKAlg and stubs in a static IV
// so that encryption tests can be guaranteed to be consistent.
type cdgWithStaticTestIV struct {
	IV []byte
	CipherDataGeneratorWithCEKAlg
}

// isAWSFixture will avoid the warning log message when doing tests that need to mock the IV
func (k cdgWithStaticTestIV) isAWSFixture() bool {
	return true
}

func (k cdgWithStaticTestIV) GenerateCipherDataWithCEKAlg(ctx aws.Context, keySize, ivSize int, cekAlg string) (CipherData, error) {
	cipherData, err := k.CipherDataGeneratorWithCEKAlg.GenerateCipherDataWithCEKAlg(ctx, keySize, ivSize, cekAlg)
	if err == nil {
		cipherData.IV = k.IV
	}
	return cipherData, err
}

func TestEncryptionClientV2_PutObject_KMSCONTEXT_AESGCM(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		fmt.Fprintln(writer, `{"CiphertextBlob":"8gSzlk7giyfFbLPUVgoVjvQebI1827jp8lDkO+n2chsiSoegx1sjm8NdPk0Bl70I","KeyId":"test-key-id","Plaintext":"lP6AbIQTmptyb/+WQq+ubDw+w7na0T1LGSByZGuaono="}`)
	}))

	sess := unit.Session.Copy()
	kmsClient := kms.New(sess.Copy(&aws.Config{Endpoint: &ts.URL}))

	var md MaterialDescription
	iv, _ := hex.DecodeString("ae325acae2bfd5b9c3d0b813")
	kmsWithStaticIV := cdgWithStaticTestIV{
		IV:                            iv,
		CipherDataGeneratorWithCEKAlg: NewKMSContextKeyGenerator(kmsClient, "test-key-id", md),
	}
	contentCipherBuilderV2 := AESGCMContentCipherBuilderV2(kmsWithStaticIV)
	client, err := NewEncryptionClientV2(sess, contentCipherBuilderV2)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	req, _ := client.PutObjectRequest(&s3.PutObjectInput{
		Bucket: aws.String("test-bucket"),
		Key:    aws.String("test-key"),
		Body: func() io.ReadSeeker {
			content, _ := hex.DecodeString("8f2c59c6dbfcacf356f3da40788cbde67ca38161a4702cbcf757af663e1c24a600001b2f500417dbf5a050f57db6737422b2ed6a44c75e0d")
			return bytes.NewReader(content)
		}(),
	})

	req.Handlers.Send.Clear()
	req.Handlers.Send.PushFront(func(r *request.Request) {
		all, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		expected, _ := hex.DecodeString("4cd8e95a1c9b8b19640e02838b02c8c09e66250703a602956695afbc23cbb8647d51645955ab63b89733d0766f9a264adb88571b1d467b734ff72eb73d31de9a83670d59688c54ea")

		if !bytes.Equal(all, expected) {
			t.Error("encrypted bytes did not match expected")
		}

		req.HTTPResponse = &http.Response{
			Status:     http.StatusText(200),
			StatusCode: http.StatusOK,
			Body:       aws.ReadSeekCloser(bytes.NewReader([]byte{})),
		}
	})
	err = req.Send()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestNewEncryptionClientV2_FailsOnIncompatibleFixtures(t *testing.T) {
	sess := unit.Session.Copy()
	_, err := NewEncryptionClientV2(sess, AESGCMContentCipherBuilder(NewKMSKeyGenerator(kms.New(sess), "cmkId")))
	if err == nil {
		t.Fatal("expected to fail, but got nil")
	}
	if !strings.Contains(err.Error(), "attempted to use deprecated or incompatible cipher builder") {
		t.Errorf("expected to get error for using dperecated cipher builder")
	}
}
