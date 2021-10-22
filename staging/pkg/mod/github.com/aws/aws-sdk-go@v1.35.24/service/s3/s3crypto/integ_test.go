// +build go1.9,s3crypto_integ

package s3crypto_test

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/integration"
	"github.com/aws/aws-sdk-go/service/kms"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

func TestInteg_EncryptFixtures(t *testing.T) {
	sess := integration.SessionWithDefaultRegion("us-west-2")

	const bucket = "aws-s3-shared-tests"
	const version = "version_2"

	cases := []struct {
		CEKAlg           string
		KEK, V1, V2, CEK string
	}{
		{
			CEKAlg: "aes_gcm",
			KEK:    "kms", V1: "AWS_SDK_TEST_ALIAS", V2: "us-west-2", CEK: "aes_gcm",
		},
		{
			CEKAlg: "aes_cbc",
			KEK:    "kms", V1: "AWS_SDK_TEST_ALIAS", V2: "us-west-2", CEK: "aes_cbc",
		},
	}

	for _, c := range cases {
		t.Run(c.CEKAlg, func(t *testing.T) {
			s3Client := s3.New(sess)

			fixtures := getFixtures(t, s3Client, c.CEKAlg, bucket)
			builder, masterKey := getEncryptFixtureBuilder(t, c.KEK, c.V1, c.V2, c.CEK)

			encClient := s3crypto.NewEncryptionClient(sess, builder)

			for caseKey, plaintext := range fixtures.Plaintexts {
				_, err := encClient.PutObject(&s3.PutObjectInput{
					Bucket: aws.String(bucket),
					Key: aws.String(
						fmt.Sprintf("%s/%s/language_Go/ciphertext_test_case_%s",
							fixtures.BaseFolder, version, caseKey),
					),
					Body: bytes.NewReader(plaintext),
					Metadata: map[string]*string{
						"Masterkey": &masterKey,
					},
				})
				if err != nil {
					t.Fatalf("failed to upload encrypted fixture, %v", err)
				}
			}
		})
	}
}

func TestInteg_DecryptFixtures(t *testing.T) {
	sess := integration.SessionWithDefaultRegion("us-west-2")

	const bucket = "aws-s3-shared-tests"
	const version = "version_2"

	cases := []struct {
		CEKAlg string
		Lang   string
	}{
		{CEKAlg: "aes_cbc", Lang: "Go"},
		{CEKAlg: "aes_gcm", Lang: "Go"},
		{CEKAlg: "aes_cbc", Lang: "Java"},
		{CEKAlg: "aes_gcm", Lang: "Java"},
	}

	for _, c := range cases {
		t.Run(c.CEKAlg+"-"+c.Lang, func(t *testing.T) {
			decClient := s3crypto.NewDecryptionClient(sess)
			s3Client := s3.New(sess)

			fixtures := getFixtures(t, s3Client, c.CEKAlg, bucket)
			ciphertexts := decryptFixtures(t, decClient, s3Client, fixtures, bucket, c.Lang, version)

			for caseKey, ciphertext := range ciphertexts {
				if e, a := len(fixtures.Plaintexts[caseKey]), len(ciphertext); e != a {
					t.Errorf("expect %v text len, got %v", e, a)
				}
				if e, a := fixtures.Plaintexts[caseKey], ciphertext; !bytes.Equal(e, a) {
					t.Errorf("expect %v text, got %v", e, a)
				}
			}
		})
	}
}

type testFixtures struct {
	BaseFolder string
	Plaintexts map[string][]byte
}

func getFixtures(t *testing.T, s3Client *s3.S3, cekAlg, bucket string) testFixtures {
	t.Helper()

	prefix := "plaintext_test_case_"
	baseFolder := "crypto_tests/" + cekAlg

	out, err := s3Client.ListObjects(&s3.ListObjectsInput{
		Bucket: aws.String(bucket),
		Prefix: aws.String(baseFolder + "/" + prefix),
	})
	if err != nil {
		t.Fatalf("unable to list fixtures %v", err)
	}

	plaintexts := map[string][]byte{}
	for _, obj := range out.Contents {
		ptObj, err := s3Client.GetObject(&s3.GetObjectInput{
			Bucket: aws.String(bucket),
			Key:    obj.Key,
		})
		if err != nil {
			t.Fatalf("unable to get fixture object %s, %v", *obj.Key, err)
		}
		caseKey := strings.TrimPrefix(*obj.Key, baseFolder+"/"+prefix)
		plaintext, err := ioutil.ReadAll(ptObj.Body)
		if err != nil {
			t.Fatalf("unable to read fixture object %s, %v", *obj.Key, err)
		}

		plaintexts[caseKey] = plaintext
	}

	return testFixtures{
		BaseFolder: baseFolder,
		Plaintexts: plaintexts,
	}
}

func getEncryptFixtureBuilder(t *testing.T, kek, v1, v2, cek string,
) (builder s3crypto.ContentCipherBuilder, masterKey string) {
	t.Helper()

	var handler s3crypto.CipherDataGenerator
	switch kek {
	case "kms":
		arn, err := getAliasInformation(v1, v2)
		if err != nil {
			t.Fatalf("failed to get fixture alias info for %s, %v", v1, err)
		}

		masterKey = base64.StdEncoding.EncodeToString([]byte(arn))
		if err != nil {
			t.Fatalf("failed to encode alias's arn %v", err)
		}

		kmsSvc := kms.New(integration.Session, &aws.Config{
			Region: &v2,
		})
		handler = s3crypto.NewKMSKeyGenerator(kmsSvc, arn)
	default:
		t.Fatalf("unknown fixture KEK, %v", kek)
	}

	switch cek {
	case "aes_gcm":
		builder = s3crypto.AESGCMContentCipherBuilder(handler)
	case "aes_cbc":
		builder = s3crypto.AESCBCContentCipherBuilder(handler, s3crypto.AESCBCPadder)
	default:
		t.Fatalf("unknown fixture CEK, %v", cek)
	}

	return builder, masterKey
}

func getAliasInformation(alias, region string) (string, error) {
	arn := ""
	svc := kms.New(integration.Session, &aws.Config{
		Region: &region,
	})

	truncated := true
	var marker *string
	for truncated {
		out, err := svc.ListAliases(&kms.ListAliasesInput{
			Marker: marker,
		})
		if err != nil {
			return arn, err
		}
		for _, aliasEntry := range out.Aliases {
			if *aliasEntry.AliasName == "alias/"+alias {
				return *aliasEntry.AliasArn, nil
			}
		}
		truncated = *out.Truncated
		marker = out.NextMarker
	}

	return "", fmt.Errorf("kms alias %s does not exist", alias)
}

func decryptFixtures(t *testing.T, decClient *s3crypto.DecryptionClient, s3Client *s3.S3,
	fixtures testFixtures, bucket, lang, version string,
) map[string][]byte {
	t.Helper()

	prefix := "ciphertext_test_case_"
	lang = "language_" + lang

	ciphertexts := map[string][]byte{}
	for caseKey := range fixtures.Plaintexts {
		cipherKey := fixtures.BaseFolder + "/" + version + "/" + lang + "/" + prefix + caseKey

		// To get metadata for encryption key
		ctObj, err := s3Client.GetObject(&s3.GetObjectInput{
			Bucket: &bucket,
			Key:    &cipherKey,
		})
		if err != nil {
			// TODO error?
			continue
		}

		// We don't support wrap, so skip it
		if ctObj.Metadata["X-Amz-Wrap-Alg"] == nil || *ctObj.Metadata["X-Amz-Wrap-Alg"] != "kms" {
			continue
		}

		ctObj, err = decClient.GetObject(&s3.GetObjectInput{
			Bucket: &bucket,
			Key:    &cipherKey,
		})
		if err != nil {
			t.Fatalf("failed to get encrypted object %v", err)
		}

		ciphertext, err := ioutil.ReadAll(ctObj.Body)
		if err != nil {
			t.Fatalf("failed to read object data %v", err)
		}
		ciphertexts[caseKey] = ciphertext
	}

	return ciphertexts
}
