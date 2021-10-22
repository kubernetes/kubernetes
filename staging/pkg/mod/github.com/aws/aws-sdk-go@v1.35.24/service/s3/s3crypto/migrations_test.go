package s3crypto_test

import (
	"bytes"
	"fmt"
	"io/ioutil"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/kms"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

// ExampleNewEncryptionClientV2_migration00 provides a migration example for how users can migrate from the V1
// encryption client to the V2 encryption client. This example demonstrates how an application using  the `kms` key wrap
// algorithm with `AES/CBC/PKCS5Padding` can migrate their application to `kms+context` key wrapping with
// `AES/GCM/NoPadding` content encryption.
func ExampleNewEncryptionClientV2_migration00() {
	sess := session.Must(session.NewSession())
	kmsClient := kms.New(sess)
	cmkID := "1234abcd-12ab-34cd-56ef-1234567890ab"

	// Usage of NewKMSKeyGenerator (kms) key wrapping algorithm must be migrated to NewKMSContextKeyGenerator (kms+context) key wrapping algorithm
	//
	// cipherDataGenerator := s3crypto.NewKMSKeyGenerator(kmsClient, cmkID)
	cipherDataGenerator := s3crypto.NewKMSContextKeyGenerator(kmsClient, cmkID, s3crypto.MaterialDescription{})

	// Usage of AESCBCContentCipherBuilder (AES/CBC/PKCS5Padding) must be migrated to AESGCMContentCipherBuilder (AES/GCM/NoPadding)
	//
	// contentCipherBuilder := s3crypto.AESCBCContentCipherBuilder(cipherDataGenerator, s3crypto.AESCBCPadder)
	contentCipherBuilder := s3crypto.AESGCMContentCipherBuilderV2(cipherDataGenerator)

	// Construction of an encryption client should be done using NewEncryptionClientV2
	//
	// encryptionClient := s3crypto.NewEncryptionClient(sess, contentCipherBuilder)
	encryptionClient, err := s3crypto.NewEncryptionClientV2(sess, contentCipherBuilder)
	if err != nil {
		fmt.Printf("failed to construct encryption client: %v", err)
		return
	}

	_, err = encryptionClient.PutObject(&s3.PutObjectInput{
		Bucket: aws.String("your_bucket"),
		Key:    aws.String("your_key"),
		Body:   bytes.NewReader([]byte("your content")),
	})
	if err != nil {
		fmt.Printf("put object error: %v\n", err)
		return
	}
	fmt.Println("put object completed")
}

// ExampleNewEncryptionClientV2_migration01 provides a more advanced migration example for how users can
// migrate from the V1 encryption client to the V2 encryption client using more complex client construction.
func ExampleNewEncryptionClientV2_migration01() {
	sess := session.Must(session.NewSession())
	kmsClient := kms.New(sess)
	cmkID := "1234abcd-12ab-34cd-56ef-1234567890ab"

	cipherDataGenerator := s3crypto.NewKMSContextKeyGenerator(kmsClient, cmkID, s3crypto.MaterialDescription{})

	contentCipherBuilder := s3crypto.AESGCMContentCipherBuilderV2(cipherDataGenerator)

	// Overriding of the encryption client options is possible by passing in functional arguments that override the
	// provided EncryptionClientOptions.
	//
	// encryptionClient := s3crypto.NewEncryptionClient(cipherDataGenerator, contentCipherBuilder, func(o *s3crypto.EncryptionClient) {
	//	 o.S3Client = s3.New(sess, &aws.Config{Region: aws.String("us-west-2")}),
	// })
	encryptionClient, err := s3crypto.NewEncryptionClientV2(sess, contentCipherBuilder, func(o *s3crypto.EncryptionClientOptions) {
		o.S3Client = s3.New(sess, &aws.Config{Region: aws.String("us-west-2")})
	})
	if err != nil {
		fmt.Printf("failed to construct encryption client: %v", err)
		return
	}

	_, err = encryptionClient.PutObject(&s3.PutObjectInput{
		Bucket: aws.String("your_bucket"),
		Key:    aws.String("your_key"),
		Body:   bytes.NewReader([]byte("your content")),
	})
	if err != nil {
		fmt.Printf("put object error: %v\n", err)
		return
	}
	fmt.Println("put object completed")
}

// ExampleNewDecryptionClientV2_migration00 provides a migration example for how users can migrate
// from the V1 Decryption Clients to the V2 Decryption Clients.
func ExampleNewDecryptionClientV2_migration00() {
	sess := session.Must(session.NewSession())

	// Construction of an decryption client must be done using NewDecryptionClientV2
	// The V2 decryption client is able to decrypt object encrypted by the V1 client.
	//
	// decryptionClient := s3crypto.NewDecryptionClient(sess)

	// The V2 decryption client requires you to explicitly register the key wrap algorithms and content encryption algorithms
	// that you want to explicitly support decryption for.
	registry := s3crypto.NewCryptoRegistry()

	kmsClient := kms.New(sess)

	// If you need support for unwrapping data keys wrapped using the `kms` wrap algorithm you can use RegisterKMSWrapWithAnyCMK.
	// Alternatively you may use RegisterKMSWrapWithCMK if you wish to limit KMS decrypt calls to a specific CMK.
	if err := s3crypto.RegisterKMSWrapWithAnyCMK(registry, kmsClient); err != nil {
		fmt.Printf("error: %v", err)
		return
	}

	// For unwrapping data keys wrapped using the new `kms+context` key wrap algorithm you can use RegisterKMSContextWrapWithAnyCMK.
	// Alternatively you may use RegisterKMSWrapWithCMK if you wish to limit KMS decrypt calls to a specific CMK.
	if err := s3crypto.RegisterKMSContextWrapWithAnyCMK(registry, kmsClient); err != nil {
		fmt.Printf("error: %v", err)
		return
	}

	// If you need to decrypt objects encrypted using the V1 AES/CBC/PCKS5Padding cipher you can do so with RegisterAESCBCContentCipher
	if err := s3crypto.RegisterAESCBCContentCipher(registry, s3crypto.AESCBCPadder); err != nil {
		fmt.Printf("error: %v", err)
		return
	}

	// For decrypting objects encrypted in V1 or V2 using AES/GCM/NoPadding cipher you can do so with RegisterAESGCMContentCipher.
	if err := s3crypto.RegisterAESGCMContentCipher(registry); err != nil {
		fmt.Printf("error: %v", err)
		return
	}

	// Instantiate a new decryption client, and provided the Wrap, cek, and Padder that have been registered
	// with your desired algorithms.
	decryptionClient, err := s3crypto.NewDecryptionClientV2(sess, registry)
	if err != nil {
		fmt.Printf("error: %v", err)
		return
	}

	getObject, err := decryptionClient.GetObject(&s3.GetObjectInput{
		Bucket: aws.String("your_bucket"),
		Key:    aws.String("your_key"),
	})
	if err != nil {
		fmt.Printf("get object error: %v\n", err)
		return
	}

	_, err = ioutil.ReadAll(getObject.Body)
	if err != nil {
		fmt.Printf("error reading object: %v\n", err)
	}
	fmt.Println("get object completed")
}

// ExampleNewDecryptionClientV2_migration01 provides a more advanced migration example for how users can
// migrate from the V1 decryption client to the V2 decryption client using more complex client construction.
func ExampleNewDecryptionClientV2_migration01() {
	sess := session.Must(session.NewSession())

	// Construction of an decryption client must be done using NewDecryptionClientV2
	// The V2 decryption client is able to decrypt object encrypted by the V1 client.
	//
	// decryptionClient := s3crypto.NewDecryptionClient(sess, func(o *s3crypto.DecryptionClient) {
	//	 o.S3Client = s3.New(sess, &aws.Config{Region: aws.String("us-west-2")})
	//})
	registry := s3crypto.NewCryptoRegistry()

	kmsClient := kms.New(sess)
	if err := s3crypto.RegisterKMSWrapWithAnyCMK(registry, kmsClient); err != nil {
		fmt.Printf("error: %v", err)
		return
	}

	// If you need to decrypt objects encrypted using AES/GCM/NoPadding cipher you can do so with RegisterAESGCMContentCipher
	if err := s3crypto.RegisterAESGCMContentCipher(registry); err != nil {
		fmt.Printf("error: %v", err)
		return
	}

	decryptionClient, err := s3crypto.NewDecryptionClientV2(sess, registry, func(o *s3crypto.DecryptionClientOptions) {
		o.S3Client = s3.New(sess, &aws.Config{Region: aws.String("us-west-2")})
	})
	if err != nil {
		fmt.Printf("error: %v", err)
		return
	}

	getObject, err := decryptionClient.GetObject(&s3.GetObjectInput{
		Bucket: aws.String("your_bucket"),
		Key:    aws.String("your_key"),
	})
	if err != nil {
		fmt.Printf("get object error: %v\n", err)
		return
	}

	_, err = ioutil.ReadAll(getObject.Body)
	if err != nil {
		fmt.Printf("error reading object: %v\n", err)
	}
	fmt.Println("get object completed")
}
