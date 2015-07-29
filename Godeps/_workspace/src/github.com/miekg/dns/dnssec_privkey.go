package dns

import (
	"crypto"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/rsa"
	"math/big"
	"strconv"
)

const format = "Private-key-format: v1.3\n"

// PrivateKey ... TODO(miek)
type PrivateKey interface {
	Sign([]byte, uint8) ([]byte, error)
	String(uint8) string
}

// PrivateKeyString converts a PrivateKey to a string. This string has the same
// format as the private-key-file of BIND9 (Private-key-format: v1.3).
// It needs some info from the key (the algorithm), so its a method of the
// DNSKEY and calls PrivateKey.String(alg).
func (r *DNSKEY) PrivateKeyString(p PrivateKey) string {
	return p.String(r.Algorithm)
}

type RSAPrivateKey rsa.PrivateKey

func (p *RSAPrivateKey) Sign(hashed []byte, alg uint8) ([]byte, error) {
	var hash crypto.Hash
	switch alg {
	case RSASHA1, RSASHA1NSEC3SHA1:
		hash = crypto.SHA1
	case RSASHA256:
		hash = crypto.SHA256
	case RSASHA512:
		hash = crypto.SHA512
	default:
		return nil, ErrAlg
	}
	return rsa.SignPKCS1v15(nil, (*rsa.PrivateKey)(p), hash, hashed)
}

func (p *RSAPrivateKey) String(alg uint8) string {
	algorithm := strconv.Itoa(int(alg)) + " (" + AlgorithmToString[alg] + ")"
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
}

type ECDSAPrivateKey ecdsa.PrivateKey

func (p *ECDSAPrivateKey) Sign(hashed []byte, alg uint8) ([]byte, error) {
	var intlen int
	switch alg {
	case ECDSAP256SHA256:
		intlen = 32
	case ECDSAP384SHA384:
		intlen = 48
	default:
		return nil, ErrAlg
	}
	r1, s1, err := ecdsa.Sign(rand.Reader, (*ecdsa.PrivateKey)(p), hashed)
	if err != nil {
		return nil, err
	}
	signature := intToBytes(r1, intlen)
	signature = append(signature, intToBytes(s1, intlen)...)
	return signature, nil
}

func (p *ECDSAPrivateKey) String(alg uint8) string {
	algorithm := strconv.Itoa(int(alg)) + " (" + AlgorithmToString[alg] + ")"
	var intlen int
	switch alg {
	case ECDSAP256SHA256:
		intlen = 32
	case ECDSAP384SHA384:
		intlen = 48
	}
	private := toBase64(intToBytes(p.D, intlen))
	return format +
		"Algorithm: " + algorithm + "\n" +
		"PrivateKey: " + private + "\n"
}

type DSAPrivateKey dsa.PrivateKey

func (p *DSAPrivateKey) Sign(hashed []byte, alg uint8) ([]byte, error) {
	r1, s1, err := dsa.Sign(rand.Reader, (*dsa.PrivateKey)(p), hashed)
	if err != nil {
		return nil, err
	}
	t := divRoundUp(divRoundUp(p.PublicKey.Y.BitLen(), 8)-64, 8)
	signature := []byte{byte(t)}
	signature = append(signature, intToBytes(r1, 20)...)
	signature = append(signature, intToBytes(s1, 20)...)
	return signature, nil
}

func (p *DSAPrivateKey) String(alg uint8) string {
	algorithm := strconv.Itoa(int(alg)) + " (" + AlgorithmToString[alg] + ")"
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
}
