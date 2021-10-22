package s3crypto

import "fmt"

type clientVersion int

const (
	v1ClientVersion clientVersion = 1 + iota
	v2ClientVersion
)

var errDeprecatedIncompatibleCipherBuilder = fmt.Errorf("attempted to use deprecated or incompatible cipher builder")

// compatibleEncryptionFixture is an unexported interface to expose whether a given fixture is compatible for encryption
// given the provided client version.
type compatibleEncryptionFixture interface {
	isEncryptionVersionCompatible(clientVersion) error
}

// awsFixture is an unexported interface to expose whether a given fixture is an aws provided fixture, and whether that
// fixtures dependencies were constructed using aws types.
//
// This interface is used in v2 clients to warn users if they are using custom implementations of ContentCipherBuilder
// or CipherDataGenerator.
type awsFixture interface {
	isAWSFixture() bool
}
