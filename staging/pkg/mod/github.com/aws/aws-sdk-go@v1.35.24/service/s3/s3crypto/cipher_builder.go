package s3crypto

import (
	"io"

	"github.com/aws/aws-sdk-go/aws"
)

// ContentCipherBuilder is a builder interface that builds
// ciphers for each request.
type ContentCipherBuilder interface {
	ContentCipher() (ContentCipher, error)
}

// ContentCipherBuilderWithContext is a builder interface that builds
// ciphers for each request.
type ContentCipherBuilderWithContext interface {
	ContentCipherWithContext(aws.Context) (ContentCipher, error)
}

// ContentCipher deals with encrypting and decrypting content
type ContentCipher interface {
	EncryptContents(io.Reader) (io.Reader, error)
	DecryptContents(io.ReadCloser) (io.ReadCloser, error)
	GetCipherData() CipherData
}

// CipherData is used for content encryption. It is used for storing the
// metadata of the encrypted content.
type CipherData struct {
	Key                 []byte
	IV                  []byte
	WrapAlgorithm       string
	CEKAlgorithm        string
	TagLength           string
	MaterialDescription MaterialDescription
	// EncryptedKey should be populated when calling GenerateCipherData
	EncryptedKey []byte

	Padder Padder
}

// Clone returns a new copy of CipherData
func (cd CipherData) Clone() (v CipherData) {
	v = cd
	v.MaterialDescription = cd.MaterialDescription.Clone()
	return v
}
