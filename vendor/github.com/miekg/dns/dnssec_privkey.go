package dns

import (
	"crypto"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/rsa"
	"math/big"
	"strconv"

	"golang.org/x/crypto/ed25519"
)

const format = "Private-key-format: v1.3\n"

// PrivateKeyString converts a PrivateKey to a string. This string has the same
// format as the private-key-file of BIND9 (Private-key-format: v1.3).
// It needs some info from the key (the algorithm), so its a method of the DNSKEY
// It supports rsa.PrivateKey, ecdsa.PrivateKey and dsa.PrivateKey
func (r *DNSKEY) PrivateKeyString(p crypto.PrivateKey) string {
	algorithm := strconv.Itoa(int(r.Algorithm))
	algorithm += " (" + AlgorithmToString[r.Algorithm] + ")"

	switch p := p.(type) {
	case *rsa.PrivateKey:
		modulus := toBase64(p.PublicKey.N.Bytes())
		e := big.NewInt(int64(p.PublicKey.E))
		publicExponent := toBase64(e.Bytes())
		privateExponent := toBase64(p.D.Bytes())
		prime1 := toBase64(p.Primes[0].Bytes())
		prime2 := toBase64(p.Primes[1].Bytes())
		// Calculate Exponent1/2 and Coefficient as per: http://en.wikipedia.org/wiki/RSA#Using_the_Chinese_remainder_algorithm
		// and from: http://code.google.com/p/go/issues/detail?id=987
		one := big.NewInt(1)
		p1 := big.NewInt(0).Sub(p.Primes[0], one)
		q1 := big.NewInt(0).Sub(p.Primes[1], one)
		exp1 := big.NewInt(0).Mod(p.D, p1)
		exp2 := big.NewInt(0).Mod(p.D, q1)
		coeff := big.NewInt(0).ModInverse(p.Primes[1], p.Primes[0])

		exponent1 := toBase64(exp1.Bytes())
		exponent2 := toBase64(exp2.Bytes())
		coefficient := toBase64(coeff.Bytes())

		return format +
			"Algorithm: " + algorithm + "\n" +
			"Modulus: " + modulus + "\n" +
			"PublicExponent: " + publicExponent + "\n" +
			"PrivateExponent: " + privateExponent + "\n" +
			"Prime1: " + prime1 + "\n" +
			"Prime2: " + prime2 + "\n" +
			"Exponent1: " + exponent1 + "\n" +
			"Exponent2: " + exponent2 + "\n" +
			"Coefficient: " + coefficient + "\n"

	case *ecdsa.PrivateKey:
		var intlen int
		switch r.Algorithm {
		case ECDSAP256SHA256:
			intlen = 32
		case ECDSAP384SHA384:
			intlen = 48
		}
		private := toBase64(intToBytes(p.D, intlen))
		return format +
			"Algorithm: " + algorithm + "\n" +
			"PrivateKey: " + private + "\n"

	case *dsa.PrivateKey:
		T := divRoundUp(divRoundUp(p.PublicKey.Parameters.G.BitLen(), 8)-64, 8)
		prime := toBase64(intToBytes(p.PublicKey.Parameters.P, 64+T*8))
		subprime := toBase64(intToBytes(p.PublicKey.Parameters.Q, 20))
		base := toBase64(intToBytes(p.PublicKey.Parameters.G, 64+T*8))
		priv := toBase64(intToBytes(p.X, 20))
		pub := toBase64(intToBytes(p.PublicKey.Y, 64+T*8))
		return format +
			"Algorithm: " + algorithm + "\n" +
			"Prime(p): " + prime + "\n" +
			"Subprime(q): " + subprime + "\n" +
			"Base(g): " + base + "\n" +
			"Private_value(x): " + priv + "\n" +
			"Public_value(y): " + pub + "\n"

	case ed25519.PrivateKey:
		private := toBase64(p.Seed())
		return format +
			"Algorithm: " + algorithm + "\n" +
			"PrivateKey: " + private + "\n"

	default:
		return ""
	}
}
