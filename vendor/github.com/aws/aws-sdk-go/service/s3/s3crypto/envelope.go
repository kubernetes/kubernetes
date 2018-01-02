package s3crypto

// DefaultInstructionKeySuffix is appended to the end of the instruction file key when
// grabbing or saving to S3
const DefaultInstructionKeySuffix = ".instruction"

const (
	metaHeader                     = "x-amz-meta"
	keyV1Header                    = "x-amz-key"
	keyV2Header                    = keyV1Header + "-v2"
	ivHeader                       = "x-amz-iv"
	matDescHeader                  = "x-amz-matdesc"
	cekAlgorithmHeader             = "x-amz-cek-alg"
	wrapAlgorithmHeader            = "x-amz-wrap-alg"
	tagLengthHeader                = "x-amz-tag-len"
	unencryptedMD5Header           = "x-amz-unencrypted-content-md5"
	unencryptedContentLengthHeader = "x-amz-unencrypted-content-length"
)

// Envelope encryption starts off by generating a random symmetric key using
// AES GCM. The SDK generates a random IV based off the encryption cipher
// chosen. The master key that was provided, whether by the user or KMS, will be used
// to encrypt the randomly generated symmetric key and base64 encode the iv. This will
// allow for decryption of that same data later.
type Envelope struct {
	// IV is the randomly generated IV base64 encoded.
	IV string `json:"x-amz-iv"`
	// CipherKey is the randomly generated cipher key.
	CipherKey string `json:"x-amz-key-v2"`
	// MaterialDesc is a description to distinguish from other envelopes.
	MatDesc               string `json:"x-amz-matdesc"`
	WrapAlg               string `json:"x-amz-wrap-alg"`
	CEKAlg                string `json:"x-amz-cek-alg"`
	TagLen                string `json:"x-amz-tag-len"`
	UnencryptedMD5        string `json:"x-amz-unencrypted-content-md5"`
	UnencryptedContentLen string `json:"x-amz-unencrypted-content-length"`
}
