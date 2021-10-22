package s3crypto

import "fmt"

var errNilCryptoRegistry = fmt.Errorf("provided CryptoRegistry must not be nil")
var errNilWrapEntry = fmt.Errorf("wrap entry must not be nil")
var errNilCEKEntry = fmt.Errorf("cek entry must not be nil")
var errNilPadder = fmt.Errorf("padder must not be nil")

func newErrDuplicateWrapEntry(name string) error {
	return newErrDuplicateRegistryEntry("wrap", name)
}

func newErrDuplicateCEKEntry(name string) error {
	return newErrDuplicateRegistryEntry("cek", name)
}

func newErrDuplicatePadderEntry(name string) error {
	return newErrDuplicateRegistryEntry("padder", name)
}

func newErrDuplicateRegistryEntry(registry, key string) error {
	return fmt.Errorf("duplicate %v registry entry, %v", registry, key)
}
