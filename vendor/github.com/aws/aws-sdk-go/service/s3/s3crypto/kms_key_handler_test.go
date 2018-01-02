package s3crypto

import (
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/kms"
)

func TestBuildKMSEncryptHandler(t *testing.T) {
	svc := kms.New(unit.Session)
	handler := NewKMSKeyGenerator(svc, "testid")
	if handler == nil {
		t.Error("expected non-nil handler")
	}
}

func TestBuildKMSEncryptHandlerWithMatDesc(t *testing.T) {
	svc := kms.New(unit.Session)
	handler := NewKMSKeyGeneratorWithMatDesc(svc, "testid", MaterialDescription{
		"Testing": aws.String("123"),
	})
	if handler == nil {
		t.Error("expected non-nil handler")
	}

	kmsHandler := handler.(*kmsKeyHandler)
	expected := MaterialDescription{
		"kms_cmk_id": aws.String("testid"),
		"Testing":    aws.String("123"),
	}

	if !reflect.DeepEqual(expected, kmsHandler.CipherData.MaterialDescription) {
		t.Errorf("expected %v, but received %v", expected, kmsHandler.CipherData.MaterialDescription)
	}
}

func TestKMSGenerateCipherData(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"CiphertextBlob":"AQEDAHhqBCCY1MSimw8gOGcUma79cn4ANvTtQyv9iuBdbcEF1QAAAH4wfAYJKoZIhvcNAQcGoG8wbQIBADBoBgkqhkiG9w0BBwEwHgYJYIZIAWUDBAEuMBEEDJ6IcN5E4wVbk38MNAIBEIA7oF1E3lS7FY9DkoxPc/UmJsEwHzL82zMqoLwXIvi8LQHr8If4Lv6zKqY8u0+JRgSVoqCvZDx3p8Cn6nM=","KeyId":"arn:aws:kms:us-west-2:042062605278:key/c80a5cdb-8d09-4f9f-89ee-df01b2e3870a","Plaintext":"6tmyz9JLBE2yIuU7iXpArqpDVle172WSmxjcO6GNT7E="}`)
	}))

	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		Endpoint:         aws.String(ts.URL),
		DisableSSL:       aws.Bool(true),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})

	svc := kms.New(sess)
	handler := NewKMSKeyGenerator(svc, "testid")

	keySize := 32
	ivSize := 16

	cd, err := handler.GenerateCipherData(keySize, ivSize)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if keySize != len(cd.Key) {
		t.Errorf("expected %d, but received %d", keySize, len(cd.Key))
	}
	if ivSize != len(cd.IV) {
		t.Errorf("expected %d, but received %d", ivSize, len(cd.IV))
	}
}

func TestKMSDecrypt(t *testing.T) {
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	keyB64 := base64.URLEncoding.EncodeToString(key)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, fmt.Sprintf("%s%s%s", `{"KeyId":"test-key-id","Plaintext":"`, keyB64, `"}`))
	}))

	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		Endpoint:         aws.String(ts.URL),
		DisableSSL:       aws.Bool(true),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})
	handler, err := (kmsKeyHandler{kms: kms.New(sess)}).decryptHandler(Envelope{MatDesc: `{"kms_cmk_id":"test"}`})
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	plaintextKey, err := handler.DecryptKey([]byte{1, 2, 3, 4})
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if !bytes.Equal(key, plaintextKey) {
		t.Errorf("expected %v, but received %v", key, plaintextKey)
	}
}

func TestKMSDecryptBadJSON(t *testing.T) {
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	keyB64 := base64.URLEncoding.EncodeToString(key)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, fmt.Sprintf("%s%s%s", `{"KeyId":"test-key-id","Plaintext":"`, keyB64, `"}`))
	}))

	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		Endpoint:         aws.String(ts.URL),
		DisableSSL:       aws.Bool(true),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})

	_, err := (kmsKeyHandler{kms: kms.New(sess)}).decryptHandler(Envelope{MatDesc: `{"kms_cmk_id":"test"`})
	if err == nil {
		t.Errorf("expected error, but received none")
	}
}
