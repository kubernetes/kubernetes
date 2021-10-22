package s3crypto

import (
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/kms"
)

func TestWrapFactory(t *testing.T) {
	o := DecryptionClientOptions{
		CryptoRegistry: initCryptoRegistryFrom(map[string]WrapEntry{
			KMSWrap: (kmsKeyHandler{
				kms: kms.New(unit.Session),
			}).decryptHandler,
		}, map[string]CEKEntry{
			AESGCMNoPadding: newAESGCMContentCipher,
		}, map[string]Padder{}),
	}
	env := Envelope{
		WrapAlg: KMSWrap,
		MatDesc: `{"kms_cmk_id":""}`,
	}
	wrap, err := wrapFromEnvelope(o, env)
	w, ok := wrap.(*kmsKeyHandler)

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if wrap == nil {
		t.Error("expected non-nil value")
	}
	if !ok {
		t.Errorf("expected kmsKeyHandler, but received %v", *w)
	}
}
func TestWrapFactoryErrorNoWrap(t *testing.T) {
	o := DecryptionClientOptions{
		CryptoRegistry: initCryptoRegistryFrom(map[string]WrapEntry{
			KMSWrap: (kmsKeyHandler{
				kms: kms.New(unit.Session),
			}).decryptHandler,
		}, map[string]CEKEntry{
			AESGCMNoPadding: newAESGCMContentCipher,
		}, map[string]Padder{}),
	}
	env := Envelope{
		WrapAlg: "none",
		MatDesc: `{"kms_cmk_id":""}`,
	}
	wrap, err := wrapFromEnvelope(o, env)

	if err == nil {
		t.Error("expected error, but received none")
	}
	if wrap != nil {
		t.Errorf("expected nil wrap value, received %v", wrap)
	}
}

func TestWrapFactoryCustomEntry(t *testing.T) {
	o := DecryptionClientOptions{
		CryptoRegistry: initCryptoRegistryFrom(map[string]WrapEntry{
			"custom": (kmsKeyHandler{
				kms: kms.New(unit.Session),
			}).decryptHandler,
		}, map[string]CEKEntry{
			AESGCMNoPadding: newAESGCMContentCipher,
		}, map[string]Padder{}),
	}
	env := Envelope{
		WrapAlg: "custom",
		MatDesc: `{"kms_cmk_id":""}`,
	}
	wrap, err := wrapFromEnvelope(o, env)

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if wrap == nil {
		t.Errorf("expected nil wrap value, received %v", wrap)
	}
}

func TestCEKFactory(t *testing.T) {
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	keyB64 := base64.URLEncoding.EncodeToString(key)
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

	o := DecryptionClientOptions{
		CryptoRegistry: initCryptoRegistryFrom(map[string]WrapEntry{
			KMSWrap: (kmsKeyHandler{
				kms: kms.New(sess),
			}).decryptHandler,
		}, map[string]CEKEntry{
			AESGCMNoPadding: newAESGCMContentCipher,
		}, map[string]Padder{
			NoPadder.Name(): NoPadder,
		}),
	}
	iv, err := hex.DecodeString("0d18e06c7c725ac9e362e1ce")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	ivB64 := base64.URLEncoding.EncodeToString(iv)

	cipherKey, err := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	cipherKeyB64 := base64.URLEncoding.EncodeToString(cipherKey)

	env := Envelope{
		WrapAlg:   KMSWrap,
		CEKAlg:    AESGCMNoPadding,
		CipherKey: cipherKeyB64,
		IV:        ivB64,
		MatDesc:   `{"kms_cmk_id":""}`,
	}
	wrap, err := wrapFromEnvelope(o, env)
	cek, err := cekFromEnvelope(o, aws.BackgroundContext(), env, wrap)

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if cek == nil {
		t.Errorf("expected non-nil cek")
	}
}

func TestCEKFactoryNoCEK(t *testing.T) {
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	keyB64 := base64.URLEncoding.EncodeToString(key)
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

	o := DecryptionClientOptions{
		CryptoRegistry: initCryptoRegistryFrom(
			map[string]WrapEntry{
				KMSWrap: (kmsKeyHandler{
					kms: kms.New(sess),
				}).decryptHandler,
			},
			map[string]CEKEntry{
				AESGCMNoPadding: newAESGCMContentCipher,
			},
			map[string]Padder{
				NoPadder.Name(): NoPadder,
			}),
	}
	iv, err := hex.DecodeString("0d18e06c7c725ac9e362e1ce")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	ivB64 := base64.URLEncoding.EncodeToString(iv)

	cipherKey, err := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	cipherKeyB64 := base64.URLEncoding.EncodeToString(cipherKey)

	env := Envelope{
		WrapAlg:   KMSWrap,
		CEKAlg:    "none",
		CipherKey: cipherKeyB64,
		IV:        ivB64,
		MatDesc:   `{"kms_cmk_id":""}`,
	}
	wrap, err := wrapFromEnvelope(o, env)
	cek, err := cekFromEnvelope(o, aws.BackgroundContext(), env, wrap)

	if err == nil {
		t.Error("expected error, but received none")
	}
	if cek != nil {
		t.Errorf("expected nil cek value, received %v", wrap)
	}
}

func TestCEKFactoryCustomEntry(t *testing.T) {
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	keyB64 := base64.URLEncoding.EncodeToString(key)
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

	o := DecryptionClientOptions{
		CryptoRegistry: initCryptoRegistryFrom(
			map[string]WrapEntry{
				KMSWrap: (kmsKeyHandler{
					kms: kms.New(sess),
				}).decryptHandler,
			}, map[string]CEKEntry{
				"custom": newAESGCMContentCipher,
			}, map[string]Padder{}),
	}
	iv, err := hex.DecodeString("0d18e06c7c725ac9e362e1ce")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	ivB64 := base64.URLEncoding.EncodeToString(iv)

	cipherKey, err := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	cipherKeyB64 := base64.URLEncoding.EncodeToString(cipherKey)

	env := Envelope{
		WrapAlg:   KMSWrap,
		CEKAlg:    "custom",
		CipherKey: cipherKeyB64,
		IV:        ivB64,
		MatDesc:   `{"kms_cmk_id":""}`,
	}
	wrap, err := wrapFromEnvelope(o, env)
	cek, err := cekFromEnvelope(o, aws.BackgroundContext(), env, wrap)

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if cek == nil {
		t.Errorf("expected non-nil cek")
	}
}
