// +build integration,go1.14

package integration

import (
	"bytes"
	"crypto/rand"
	"flag"
	"io"
	"io/ioutil"
	"log"
	"os"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/awstesting/integration"
	"github.com/aws/aws-sdk-go/service/kms"
	"github.com/aws/aws-sdk-go/service/kms/kmsiface"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

var config = &struct {
	Enabled  bool
	Region   string
	KMSKeyID string
	Bucket   string
	Session  *session.Session
	Clients  struct {
		KMS kmsiface.KMSAPI
		S3  s3iface.S3API
	}
}{}

func init() {
	flag.BoolVar(&config.Enabled, "enable", false, "enable integration testing")
	flag.StringVar(&config.Region, "region", "us-west-2", "integration test region")
	flag.StringVar(&config.KMSKeyID, "kms-key-id", "", "KMS CMK Key ID")
	flag.StringVar(&config.Bucket, "bucket", "", "S3 Bucket Name")
}

func TestMain(m *testing.M) {
	flag.Parse()
	if !config.Enabled {
		log.Println("skipping s3crypto integration tests")
		os.Exit(0)
	}

	if len(config.Bucket) == 0 {
		log.Fatal("bucket name must be provided")
	}

	if len(config.KMSKeyID) == 0 {
		log.Fatal("kms cmk key id must be provided")
	}

	config.Session = session.Must(session.NewSession(&aws.Config{Region: &config.Region}))

	config.Clients.KMS = kms.New(config.Session)
	config.Clients.S3 = s3.New(config.Session)

	m.Run()
}

func TestEncryptionV1_WithV2Interop(t *testing.T) {
	kmsKeyGenerator := s3crypto.NewKMSKeyGenerator(config.Clients.KMS, config.KMSKeyID)

	// 1020 is chosen here as it is not cleanly divisible by the AES-256 block size
	testData := make([]byte, 1020)
	_, err := rand.Read(testData)
	if err != nil {
		t.Fatalf("failed to read random data: %v", err)
	}

	v1DC := s3crypto.NewDecryptionClient(config.Session, func(client *s3crypto.DecryptionClient) {
		client.S3Client = config.Clients.S3
	})

	cr := s3crypto.NewCryptoRegistry()
	if err = s3crypto.RegisterKMSWrapWithAnyCMK(cr, config.Clients.KMS); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if err = s3crypto.RegisterKMSContextWrapWithAnyCMK(cr, config.Clients.KMS); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if err = s3crypto.RegisterAESGCMContentCipher(cr); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if err = s3crypto.RegisterAESCBCContentCipher(cr, s3crypto.AESCBCPadder); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	v2DC, err := s3crypto.NewDecryptionClientV2(config.Session, cr, func(options *s3crypto.DecryptionClientOptions) {
		options.S3Client = config.Clients.S3
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	cases := map[string]s3crypto.ContentCipherBuilder{
		"AES/GCM/NoPadding":    s3crypto.AESGCMContentCipherBuilder(kmsKeyGenerator),
		"AES/CBC/PKCS5Padding": s3crypto.AESCBCContentCipherBuilder(kmsKeyGenerator, s3crypto.AESCBCPadder),
	}

	for name, ccb := range cases {
		t.Run(name, func(t *testing.T) {
			ec := s3crypto.NewEncryptionClient(config.Session, ccb, func(client *s3crypto.EncryptionClient) {
				client.S3Client = config.Clients.S3
			})
			id := integration.UniqueID()
			// PutObject with V1 Client
			putObject(t, ec, id, bytes.NewReader(testData))
			// Verify V1 Decryption Client
			getObjectAndCompare(t, v1DC, id, testData)
			// Verify V2 Decryption Client
			getObjectAndCompare(t, v2DC, id, testData)
		})
	}
}

func TestEncryptionV2_WithV1Interop(t *testing.T) {
	kmsKeyGenerator := s3crypto.NewKMSContextKeyGenerator(config.Clients.KMS, config.KMSKeyID, s3crypto.MaterialDescription{})
	gcmContentCipherBuilder := s3crypto.AESGCMContentCipherBuilderV2(kmsKeyGenerator)

	ec, err := s3crypto.NewEncryptionClientV2(config.Session, gcmContentCipherBuilder, func(options *s3crypto.EncryptionClientOptions) {
		options.S3Client = config.Clients.S3
	})
	if err != nil {
		t.Fatalf("failed to construct encryption decryptionClient: %v", err)
	}

	decryptionClient := s3crypto.NewDecryptionClient(config.Session, func(client *s3crypto.DecryptionClient) {
		client.S3Client = config.Clients.S3
	})

	cr := s3crypto.NewCryptoRegistry()
	if err = s3crypto.RegisterKMSContextWrapWithAnyCMK(cr, config.Clients.KMS); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if err = s3crypto.RegisterAESGCMContentCipher(cr); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	decryptionClientV2, err := s3crypto.NewDecryptionClientV2(config.Session, cr, func(options *s3crypto.DecryptionClientOptions) {
		options.S3Client = config.Clients.S3
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// 1020 is chosen here as it is not cleanly divisible by the AES-256 block size
	testData := make([]byte, 1020)
	_, err = rand.Read(testData)
	if err != nil {
		t.Fatalf("failed to read random data: %v", err)
	}

	keyId := integration.UniqueID()

	// Upload V2 Objects with Encryption Client
	putObject(t, ec, keyId, bytes.NewReader(testData))

	// Verify V2 Object with V2 Decryption Client
	getObjectAndCompare(t, decryptionClientV2, keyId, testData)

	// Verify V2 Object with V1 Decryption Client
	getObjectAndCompare(t, decryptionClient, keyId, testData)
}

type Encryptor interface {
	PutObject(input *s3.PutObjectInput) (*s3.PutObjectOutput, error)
}

func putObject(t *testing.T, client Encryptor, key string, reader io.ReadSeeker) {
	t.Helper()
	_, err := client.PutObject(&s3.PutObjectInput{
		Bucket: &config.Bucket,
		Key:    &key,
		Body:   reader,
	})
	if err != nil {
		t.Fatalf("failed to upload object: %v", err)
	}
	t.Cleanup(doKeyCleanup(key))
}

type Decryptor interface {
	GetObject(input *s3.GetObjectInput) (*s3.GetObjectOutput, error)
}

func getObjectAndCompare(t *testing.T, client Decryptor, key string, expected []byte) {
	t.Helper()
	output, err := client.GetObject(&s3.GetObjectInput{
		Bucket: &config.Bucket,
		Key:    &key,
	})
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}

	actual, err := ioutil.ReadAll(output.Body)
	if err != nil {
		t.Fatalf("failed to read body response: %v", err)
	}

	if bytes.Compare(expected, actual) != 0 {
		t.Errorf("expected bytes did not match actual")
	}
}

func doKeyCleanup(key string) func() {
	return func() {
		_, err := config.Clients.S3.DeleteObject(&s3.DeleteObjectInput{
			Bucket: &config.Bucket,
			Key:    &key,
		})
		if err != nil {
			log.Printf("failed to delete %s: %v", key, err)
		}
	}
}
