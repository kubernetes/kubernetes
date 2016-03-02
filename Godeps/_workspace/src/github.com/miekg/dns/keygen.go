package dns

import (
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"math/big"
	"strconv"
)

const _FORMAT = "Private-key-format: v1.3\n"

// Empty interface that is used as a wrapper around all possible
// private key implementations from the crypto package.
type PrivateKey interface{}

// Generate generates a DNSKEY of the given bit size.
// The public part is put inside the DNSKEY record.
// The Algorithm in the key must be set as this will define
// what kind of DNSKEY will be generated.
// The ECDSA algorithms imply a fixed keysize, in that case
// bits should be set to the size of the algorithm.
func (r *DNSKEY) Generate(bits int) (PrivateKey, error) {
	switch r.Algorithm {
	case DSA, DSANSEC3SHA1:
		if bits != 1024 {
			return nil, ErrKeySize
		}
	case RSAMD5, RSASHA1, RSASHA256, RSASHA1NSEC3SHA1:
		if bits < 512 || bits > 4096 {
			return nil, ErrKeySize
		}
	case RSASHA512:
		if bits < 1024 || bits > 4096 {
			return nil, ErrKeySize
		}
	case ECDSAP256SHA256:
		if bits != 256 {
			return nil, ErrKeySize
		}
	case ECDSAP384SHA384:
		if bits != 384 {
			return nil, ErrKeySize
		}
	}

	switch r.Algorithm {
	case DSA, DSANSEC3SHA1:
		params := new(dsa.Parameters)
		if err := dsa.GenerateParameters(params, rand.Reader, dsa.L1024N160); err != nil {
			return nil, err
		}
		priv := new(dsa.PrivateKey)
		priv.PublicKey.Parameters = *params
		err := dsa.GenerateKey(priv, rand.Reader)
		if err != nil {
			return nil, err
		}
		r.setPublicKeyDSA(params.Q, params.P, params.G, priv.PublicKey.Y)
		return priv, nil
	case RSAMD5, RSASHA1, RSASHA256, RSASHA512, RSASHA1NSEC3SHA1:
		priv, err := rsa.GenerateKey(rand.Reader, bits)
		if err != nil {
			return nil, err
		}
		r.setPublicKeyRSA(priv.PublicKey.E, priv.PublicKey.N)
		return priv, nil
	case ECDSAP256SHA256, ECDSAP384SHA384:
		var c elliptic.Curve
		switch r.Algorithm {
		case ECDSAP256SHA256:
			c = elliptic.P256()
		case ECDSAP384SHA384:
			c = elliptic.P384()
		}
		priv, err := ecdsa.GenerateKey(c, rand.Reader)
		if err != nil {
			return nil, err
		}
		r.setPublicKeyCurve(priv.PublicKey.X, priv.PublicKey.Y)
		return priv, nil
	default:
		return nil, ErrAlg
	}
	return nil, nil // Dummy return
}

// PrivateKeyString converts a PrivateKey to a string. This
// string has the same format as the private-key-file of BIND9 (Private-key-format: v1.3).
// It needs some info from the key (hashing, keytag), so its a method of the DNSKEY.
func (r *DNSKEY) PrivateKeyString(p PrivateKey) (s string) {
	switch t := p.(type) {
	case *rsa.PrivateKey:
		algorithm := strconv.Itoa(int(r.Algorithm)) + " (" + AlgorithmToString[r.Algorithm] + ")"
		modulus := toBase64(t.PublicKey.N.Bytes())
		e := big.NewInt(int64(t.PublicKey.E))
		publicExponent := toBase64(e.Bytes())
		privateExponent := toBase64(t.D.Bytes())
		prime1 := toBase64(t.Primes[0].Bytes())
		prime2 := toBase64(t.Primes[1].Bytes())
		// Calculate Exponent1/2 and Coefficient as per: http://en.wikipedia.org/wiki/RSA#Using_the_Chinese_remainder_algorithm
		// and from: http://code.google.com/p/go/issues/detail?id=987
		one := big.NewInt(1)
		minusone := big.NewInt(-1)
		p_1 := big.NewInt(0).Sub(t.Primes[0], one)
		q_1 := big.NewInt(0).Sub(t.Primes[1], one)
		exp1 := big.NewInt(0).Mod(t.D, p_1)
		exp2 := big.NewInt(0).Mod(t.D, q_1)
		coeff := big.NewInt(0).Exp(t.Primes[1], minusone, t.Primes[0])

		exponent1 := toBase64(exp1.Bytes())
		exponent2 := toBase64(exp2.Bytes())
		coefficient := toBase64(coeff.Bytes())

		s = _FORMAT +
			"Algorithm: " + algorithm + "\n" +
			"Modules: " + modulus + "\n" +
			"PublicExponent: " + publicExponent + "\n" +
			"PrivateExponent: " + privateExponent + "\n" +
			"Prime1: " + prime1 + "\n" +
			"Prime2: " + prime2 + "\n" +
			"Exponent1: " + exponent1 + "\n" +
			"Exponent2: " + exponent2 + "\n" +
			"Coefficient: " + coefficient + "\n"
	case *ecdsa.PrivateKey:
		algorithm := strconv.Itoa(int(r.Algorithm)) + " (" + AlgorithmToString[r.Algorithm] + ")"
		var intlen int
		switch r.Algorithm {
		case ECDSAP256SHA256:
			intlen = 32
		case ECDSAP384SHA384:
			intlen = 48
		}
		private := toBase64(intToBytes(t.D, intlen))
		s = _FORMAT +
			"Algorithm: " + algorithm + "\n" +
			"PrivateKey: " + private + "\n"
	case *dsa.PrivateKey:
		algorithm := strconv.Itoa(int(r.Algorithm)) + " (" + AlgorithmToString[r.Algorithm] + ")"
		T := divRoundUp(divRoundUp(t.PublicKey.Parameters.G.BitLen(), 8)-64, 8)
		prime := toBase64(intToBytes(t.PublicKey.Parameters.P, 64+T*8))
		subprime := toBase64(intToBytes(t.PublicKey.Parameters.Q, 20))
		base := toBase64(intToBytes(t.PublicKey.Parameters.G, 64+T*8))
		priv := toBase64(intToBytes(t.X, 20))
		pub := toBase64(intToBytes(t.PublicKey.Y, 64+T*8))
		s = _FORMAT +
			"Algorithm: " + algorithm + "\n" +
			"Prime(p): " + prime + "\n" +
			"Subprime(q): " + subprime + "\n" +
			"Base(g): " + base + "\n" +
			"Private_value(x): " + priv + "\n" +
			"Public_value(y): " + pub + "\n"
	}
	return
}
