package s3crypto

import (
	"encoding/json"
	"fmt"
	"strconv"
)

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
	MatDesc string `json:"x-amz-matdesc"`
	WrapAlg string `json:"x-amz-wrap-alg"`
	CEKAlg  string `json:"x-amz-cek-alg"`
	TagLen  string `json:"x-amz-tag-len"`

	// deprecated: This MD5 hash is no longer populated
	UnencryptedMD5 string `json:"-"`

	UnencryptedContentLen string `json:"x-amz-unencrypted-content-length"`
}

// UnmarshalJSON unmarshalls the given JSON bytes into Envelope
func (e *Envelope) UnmarshalJSON(value []byte) error {
	type StrictEnvelope Envelope
	type LaxEnvelope struct {
		StrictEnvelope
		TagLen                json.RawMessage `json:"x-amz-tag-len"`
		UnencryptedContentLen json.RawMessage `json:"x-amz-unencrypted-content-length"`
	}

	inner := LaxEnvelope{}
	err := json.Unmarshal(value, &inner)
	if err != nil {
		return err
	}
	*e = Envelope(inner.StrictEnvelope)

	e.TagLen, err = getJSONNumberAsString(inner.TagLen)
	if err != nil {
		return fmt.Errorf("failed to parse tag length: %v", err)
	}

	e.UnencryptedContentLen, err = getJSONNumberAsString(inner.UnencryptedContentLen)
	if err != nil {
		return fmt.Errorf("failed to parse unencrypted content length: %v", err)
	}

	return nil
}

// getJSONNumberAsString will attempt to convert the provided bytes into a string representation of a JSON Number.
// Only supports byte values that are string or integers, not floats. If the provided value is JSON Null, empty string
// will be returned.
func getJSONNumberAsString(data []byte) (string, error) {
	if len(data) == 0 {
		return "", nil
	}

	// first try string, this also catches null value
	var s *string
	err := json.Unmarshal(data, &s)
	if err == nil && s != nil {
		return *s, nil
	} else if err == nil {
		return "", nil
	}

	// fallback to int64
	var i int64
	err = json.Unmarshal(data, &i)
	if err == nil {
		return strconv.FormatInt(i, 10), nil
	}

	return "", fmt.Errorf("failed to parse as JSON Number")
}
