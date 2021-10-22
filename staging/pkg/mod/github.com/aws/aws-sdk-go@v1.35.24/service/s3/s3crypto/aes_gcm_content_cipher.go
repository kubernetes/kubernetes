package s3crypto

import (
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go/aws"
)

const (
	gcmKeySize   = 32
	gcmNonceSize = 12
)

// AESGCMContentCipherBuilder returns a new encryption only AES/GCM mode structure with a specific cipher data generator
// that will provide keys to be used for content encryption.
//
// Note: This uses the Go stdlib AEAD implementation for AES/GCM. Due to this objects to be encrypted or decrypted
// will be fully loaded into memory before encryption or decryption can occur. Caution must be taken to avoid memory
// allocation failures.
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func AESGCMContentCipherBuilder(generator CipherDataGenerator) ContentCipherBuilder {
	return gcmContentCipherBuilder{generator}
}

// AESGCMContentCipherBuilderV2 returns a new encryption only AES/GCM mode structure with a specific cipher data generator
// that will provide keys to be used for content encryption. This type is compatible with the V2 encryption client.
//
// Note: This uses the Go stdlib AEAD implementation for AES/GCM. Due to this objects to be encrypted or decrypted
// will be fully loaded into memory before encryption or decryption can occur. Caution must be taken to avoid memory
// allocation failures.
func AESGCMContentCipherBuilderV2(generator CipherDataGeneratorWithCEKAlg) ContentCipherBuilder {
	return gcmContentCipherBuilderV2{generator}
}

// RegisterAESGCMContentCipher registers the AES/GCM content cipher algorithm with the provided CryptoRegistry.
//
// Example:
//	cr := s3crypto.NewCryptoRegistry()
//	if err := s3crypto.RegisterAESGCMContentCipher(cr); err != nil {
//		panic(err) // handle error
//	}
//
func RegisterAESGCMContentCipher(registry *CryptoRegistry) error {
	if registry == nil {
		return errNilCryptoRegistry
	}

	err := registry.AddCEK(AESGCMNoPadding, newAESGCMContentCipher)
	if err != nil {
		return err
	}

	// NoPadder is generic but required by this algorithm, so if it is already registered and is the expected implementation
	// don't error.
	padderName := NoPadder.Name()
	if v, ok := registry.GetPadder(padderName); !ok {
		if err := registry.AddPadder(padderName, NoPadder); err != nil {
			return err
		}
	} else if _, ok := v.(noPadder); !ok {
		return fmt.Errorf("%s is already registred but does not match expected type %T", padderName, NoPadder)
	}
	return nil
}

// gcmContentCipherBuilder is a AES/GCM content cipher to be used with the V1 client CipherDataGenerator interface
type gcmContentCipherBuilder struct {
	generator CipherDataGenerator
}

func (builder gcmContentCipherBuilder) ContentCipher() (ContentCipher, error) {
	return builder.ContentCipherWithContext(aws.BackgroundContext())
}

func (builder gcmContentCipherBuilder) ContentCipherWithContext(ctx aws.Context) (ContentCipher, error) {
	var cd CipherData
	var err error

	switch v := builder.generator.(type) {
	case CipherDataGeneratorWithContext:
		cd, err = v.GenerateCipherDataWithContext(ctx, gcmKeySize, gcmNonceSize)
	default:
		cd, err = builder.generator.GenerateCipherData(gcmKeySize, gcmNonceSize)
	}
	if err != nil {
		return nil, err
	}

	return newAESGCMContentCipher(cd)
}

// isFixtureEncryptionCompatible will ensure that this type may only be used with the V1 client
func (builder gcmContentCipherBuilder) isEncryptionVersionCompatible(version clientVersion) error {
	if version != v1ClientVersion {
		return errDeprecatedIncompatibleCipherBuilder
	}
	return nil
}

func (builder gcmContentCipherBuilder) isAWSFixture() bool {
	return true
}

// gcmContentCipherBuilderV2 return a new builder for encryption content using AES/GCM/NoPadding. This type is meant
// to be used with key wrapping implementations that allow the cek algorithm to be provided when calling the
// cipher data generator.
type gcmContentCipherBuilderV2 struct {
	generator CipherDataGeneratorWithCEKAlg
}

func (builder gcmContentCipherBuilderV2) ContentCipher() (ContentCipher, error) {
	return builder.ContentCipherWithContext(aws.BackgroundContext())
}

func (builder gcmContentCipherBuilderV2) ContentCipherWithContext(ctx aws.Context) (ContentCipher, error) {
	cd, err := builder.generator.GenerateCipherDataWithCEKAlg(ctx, gcmKeySize, gcmNonceSize, AESGCMNoPadding)
	if err != nil {
		return nil, err
	}

	return newAESGCMContentCipher(cd)
}

// isFixtureEncryptionCompatible will ensure that this type may only be used with the V2 client
func (builder gcmContentCipherBuilderV2) isEncryptionVersionCompatible(version clientVersion) error {
	if version != v2ClientVersion {
		return errDeprecatedIncompatibleCipherBuilder
	}
	return nil
}

// isAWSFixture will return whether this type was constructed with an AWS provided CipherDataGenerator
func (builder gcmContentCipherBuilderV2) isAWSFixture() bool {
	v, ok := builder.generator.(awsFixture)
	return ok && v.isAWSFixture()
}

func newAESGCMContentCipher(cd CipherData) (ContentCipher, error) {
	cd.CEKAlgorithm = AESGCMNoPadding
	cd.TagLength = "128"

	cipher, err := newAESGCM(cd)
	if err != nil {
		return nil, err
	}

	return &aesGCMContentCipher{
		CipherData: cd,
		Cipher:     cipher,
	}, nil
}

// AESGCMContentCipher will use AES GCM for the main cipher.
type aesGCMContentCipher struct {
	CipherData CipherData
	Cipher     Cipher
}

// EncryptContents will generate a random key and iv and encrypt the data using cbc
func (cc *aesGCMContentCipher) EncryptContents(src io.Reader) (io.Reader, error) {
	return cc.Cipher.Encrypt(src), nil
}

// DecryptContents will use the symmetric key provider to instantiate a new GCM cipher.
// We grab a decrypt reader from gcm and wrap it in a CryptoReadCloser. The only error
// expected here is when the key or iv is of invalid length.
func (cc *aesGCMContentCipher) DecryptContents(src io.ReadCloser) (io.ReadCloser, error) {
	reader := cc.Cipher.Decrypt(src)
	return &CryptoReadCloser{Body: src, Decrypter: reader}, nil
}

// GetCipherData returns cipher data
func (cc aesGCMContentCipher) GetCipherData() CipherData {
	return cc.CipherData
}

// assert ContentCipherBuilder implementations
var (
	_ ContentCipherBuilder = (*gcmContentCipherBuilder)(nil)
	_ ContentCipherBuilder = (*gcmContentCipherBuilderV2)(nil)
)

// assert ContentCipherBuilderWithContext implementations
var (
	_ ContentCipherBuilderWithContext = (*gcmContentCipherBuilder)(nil)
	_ ContentCipherBuilderWithContext = (*gcmContentCipherBuilderV2)(nil)
)

// assert ContentCipher implementations
var (
	_ ContentCipher = (*aesGCMContentCipher)(nil)
)

// assert awsFixture implementations
var (
	_ awsFixture = (*gcmContentCipherBuilderV2)(nil)
)
