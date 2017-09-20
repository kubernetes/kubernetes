package dhcp4client

import (
	cryptorand "crypto/rand"
	mathrand "math/rand"
)

func CryptoGenerateXID(b []byte) {
	if _, err := cryptorand.Read(b); err != nil {
		panic(err)
	}
}

func MathGenerateXID(b []byte) {
	if _, err := mathrand.Read(b); err != nil {
		panic(err)
	}
}
