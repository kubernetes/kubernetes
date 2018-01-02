package s3crypto

import "crypto/rand"

// CipherDataGenerator handles generating proper key and IVs of proper size for the
// content cipher. CipherDataGenerator will also encrypt the key and store it in
// the CipherData.
type CipherDataGenerator interface {
	GenerateCipherData(int, int) (CipherData, error)
}

// CipherDataDecrypter is a handler to decrypt keys from the envelope.
type CipherDataDecrypter interface {
	DecryptKey([]byte) ([]byte, error)
}

func generateBytes(n int) []byte {
	b := make([]byte, n)
	rand.Read(b)
	return b
}
