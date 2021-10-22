package s3crypto

import (
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/kms"
)

func TestKmsContextKeyHandler_GenerateCipherDataWithCEKAlg(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, err := ioutil.ReadAll(r.Body)
		if err != nil {
			w.WriteHeader(500)
			return
		}
		var body map[string]interface{}
		err = json.Unmarshal(bodyBytes, &body)
		if err != nil {
			w.WriteHeader(500)
			return
		}

		md, ok := body["EncryptionContext"].(map[string]interface{})
		if !ok {
			w.WriteHeader(500)
			return
		}

		exEncContext := map[string]interface{}{
			"aws:" + cekAlgorithmHeader: "cekAlgValue",
		}

		if e, a := exEncContext, md; !reflect.DeepEqual(e, a) {
			w.WriteHeader(500)
			t.Errorf("expected %v, got %v", e, a)
			return
		}

		fmt.Fprintln(w, `{"CiphertextBlob":"AQEDAHhqBCCY1MSimw8gOGcUma79cn4ANvTtQyv9iuBdbcEF1QAAAH4wfAYJKoZIhvcNAQcGoG8wbQIBADBoBgkqhkiG9w0BBwEwHgYJYIZIAWUDBAEuMBEEDJ6IcN5E4wVbk38MNAIBEIA7oF1E3lS7FY9DkoxPc/UmJsEwHzL82zMqoLwXIvi8LQHr8If4Lv6zKqY8u0+JRgSVoqCvZDx3p8Cn6nM=","KeyId":"arn:aws:kms:us-west-2:042062605278:key/c80a5cdb-8d09-4f9f-89ee-df01b2e3870a","Plaintext":"6tmyz9JLBE2yIuU7iXpArqpDVle172WSmxjcO6GNT7E="}`)
	}))
	defer ts.Close()

	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		Endpoint:         aws.String(ts.URL),
		DisableSSL:       aws.Bool(true),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})

	svc := kms.New(sess)
	handler := NewKMSContextKeyGenerator(svc, "testid", nil)

	keySize := 32
	ivSize := 16

	cd, err := handler.GenerateCipherDataWithCEKAlg(aws.BackgroundContext(), keySize, ivSize, "cekAlgValue")
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

func TestKmsContextKeyHandler_GenerateCipherDataWithCEKAlg_ReservedKeyConflict(t *testing.T) {
	svc := kms.New(unit.Session.Copy())
	handler := NewKMSContextKeyGenerator(svc, "testid", MaterialDescription{
		"aws:x-amz-cek-alg": aws.String("something unexpected"),
	})

	_, err := handler.GenerateCipherDataWithCEKAlg(aws.BackgroundContext(), 32, 16, "cekAlgValue")
	if err == nil {
		t.Errorf("expected error, but none")
	} else if !strings.Contains(err.Error(), "conflict in reserved KMS Encryption Context key aws:x-amz-cek-alg") {
		t.Errorf("expected reserved key error, got %v", err)
	}
}

func TestKmsContextKeyHandler_DecryptKey(t *testing.T) {
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	keyB64 := base64.URLEncoding.EncodeToString(key)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Errorf("expected no error, got %v", err)
			w.WriteHeader(500)
			return
		}

		var body map[string]interface{}
		err = json.Unmarshal(bodyBytes, &body)
		if err != nil {
			w.WriteHeader(500)
			return
		}

		if _, ok := body["KeyId"]; ok {
			t.Errorf("expected CMK to not be sent")
		}

		md, ok := body["EncryptionContext"].(map[string]interface{})
		if !ok {
			w.WriteHeader(500)
			return
		}

		exEncContext := map[string]interface{}{
			"aws:" + cekAlgorithmHeader: "AES/GCM/NoPadding",
		}

		if e, a := exEncContext, md; !reflect.DeepEqual(e, a) {
			w.WriteHeader(500)
			t.Errorf("expected %v, got %v", e, a)
			return
		}

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
	handler, err := newKMSContextWrapEntryWithAnyCMK(kms.New(sess))(Envelope{WrapAlg: KMSContextWrap, CEKAlg: "AES/GCM/NoPadding", MatDesc: `{"aws:x-amz-cek-alg": "AES/GCM/NoPadding"}`})
	if err != nil {
		t.Fatalf("expected no error, but received %v", err)
	}

	plaintextKey, err := handler.DecryptKey([]byte{1, 2, 3, 4})
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if !bytes.Equal(key, plaintextKey) {
		t.Errorf("expected %v, but received %v", key, plaintextKey)
	}
}

func TestKmsContextKeyHandler_decryptHandler_MismatchCEK(t *testing.T) {
	_, err := newKMSContextWrapEntryWithAnyCMK(kms.New(unit.Session.Copy()))(Envelope{WrapAlg: KMSContextWrap, CEKAlg: "MismatchCEKValue", MatDesc: `{"aws:x-amz-cek-alg": "AES/GCM/NoPadding"}`})
	if err == nil {
		t.Fatal("expected error, but got none")
	}

	if e, a := "algorithm used at encryption time does not match the algorithm stored", err.Error(); !strings.Contains(a, e) {
		t.Errorf("expected error to contain %v, got %v", e, a)
	}
}

func TestKmsContextKeyHandler_decryptHandler_MissingContextKey(t *testing.T) {
	_, err := newKMSContextWrapEntryWithAnyCMK(kms.New(unit.Session.Copy()))(Envelope{WrapAlg: KMSContextWrap, CEKAlg: "AES/GCM/NoPadding", MatDesc: `{}`})
	if err == nil {
		t.Fatal("expected error, but got none")
	}

	if e, a := "missing from encryption context", err.Error(); !strings.Contains(a, e) {
		t.Errorf("expected error to contain %v, got %v", e, a)
	}
}

func TestKmsContextKeyHandler_decryptHandler_MismatchWrap(t *testing.T) {
	_, err := newKMSContextWrapEntryWithAnyCMK(kms.New(unit.Session.Copy()))(Envelope{WrapAlg: KMSWrap, CEKAlg: "AES/GCM/NoPadding", MatDesc: `{}`})
	if err == nil {
		t.Fatal("expected error, but got none")
	}

	if e, a := "x-amz-cek-alg value `kms` did not match the expected algorithm `kms+context` for this handler", err.Error(); !strings.Contains(a, e) {
		t.Errorf("expected error to contain %v, got %v", e, a)
	}
}

func TestKmsContextKeyHandler_DecryptKey_WithCMK(t *testing.T) {
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	keyB64 := base64.URLEncoding.EncodeToString(key)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Errorf("expected no error, got %v", err)
			w.WriteHeader(500)
			return
		}

		if !bytes.Contains(body, []byte(`"KeyId":"thisKey"`)) {
			t.Errorf("expected CMK to be sent")
		}

		fmt.Fprintln(w, fmt.Sprintf("%s%s%s", `{"KeyId":"thisKey","Plaintext":"`, keyB64, `"}`))
	}))
	defer ts.Close()

	sess := unit.Session.Copy(&aws.Config{
		MaxRetries:       aws.Int(0),
		Endpoint:         aws.String(ts.URL),
		DisableSSL:       aws.Bool(true),
		S3ForcePathStyle: aws.Bool(true),
		Region:           aws.String("us-west-2"),
	})
	handler, err := newKMSContextWrapEntryWithCMK(kms.New(sess), "thisKey")(Envelope{WrapAlg: KMSContextWrap, CEKAlg: "AES/GCM/NoPadding", MatDesc: `{"aws:x-amz-cek-alg": "AES/GCM/NoPadding"}`})
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	_, err = handler.DecryptKey([]byte{1, 2, 3, 4})
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
}

func TestRegisterKMSContextWrapWithAnyCMK(t *testing.T) {
	kmsClient := kms.New(unit.Session.Copy())

	cr := NewCryptoRegistry()
	if err := RegisterKMSContextWrapWithAnyCMK(cr, kmsClient); err != nil {
		t.Errorf("expected no error, got %v", err)
	}

	if wrap, ok := cr.GetWrap(KMSContextWrap); !ok {
		t.Errorf("expected wrapped to be present")
	} else if wrap == nil {
		t.Errorf("expected wrap to not be nil")
	}

	if err := RegisterKMSContextWrapWithCMK(cr, kmsClient, "test-key-id"); err == nil {
		t.Error("expected error, got none")
	}
}

func TestRegisterKMSContextWrapWithCMK(t *testing.T) {
	kmsClient := kms.New(unit.Session.Copy())

	cr := NewCryptoRegistry()
	if err := RegisterKMSContextWrapWithCMK(cr, kmsClient, "cmkId"); err != nil {
		t.Errorf("expected no error, got %v", err)
	}

	if wrap, ok := cr.GetWrap(KMSContextWrap); !ok {
		t.Errorf("expected wrapped to be present")
	} else if wrap == nil {
		t.Errorf("expected wrap to not be nil")
	}

	if err := RegisterKMSContextWrapWithAnyCMK(cr, kmsClient); err == nil {
		t.Error("expected error, got none")
	}
}
