package libtrust

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
)

/*
 * RSA DSA PUBLIC KEY
 */

// rsaPublicKey implements a JWK Public Key using RSA digital signature algorithms.
type rsaPublicKey struct {
	*rsa.PublicKey
	extended map[string]interface{}
}

func fromRSAPublicKey(cryptoPublicKey *rsa.PublicKey) *rsaPublicKey {
	return &rsaPublicKey{cryptoPublicKey, map[string]interface{}{}}
}

// KeyType returns the JWK key type for RSA keys, i.e., "RSA".
func (k *rsaPublicKey) KeyType() string {
	return "RSA"
}

// KeyID returns a distinct identifier which is unique to this Public Key.
func (k *rsaPublicKey) KeyID() string {
	return keyIDFromCryptoKey(k)
}

func (k *rsaPublicKey) String() string {
	return fmt.Sprintf("RSA Public Key <%s>", k.KeyID())
}

// Verify verifyies the signature of the data in the io.Reader using this Public Key.
// The alg parameter should be the name of the JWA digital signature algorithm
// which was used to produce the signature and should be supported by this
// public key. Returns a nil error if the signature is valid.
func (k *rsaPublicKey) Verify(data io.Reader, alg string, signature []byte) error {
	// Verify the signature of the given date, return non-nil error if valid.
	sigAlg, err := rsaSignatureAlgorithmByName(alg)
	if err != nil {
		return fmt.Errorf("unable to verify Signature: %s", err)
	}

	hasher := sigAlg.HashID().New()
	_, err = io.Copy(hasher, data)
	if err != nil {
		return fmt.Errorf("error reading data to sign: %s", err)
	}
	hash := hasher.Sum(nil)

	err = rsa.VerifyPKCS1v15(k.PublicKey, sigAlg.HashID(), hash, signature)
	if err != nil {
		return fmt.Errorf("invalid %s signature: %s", sigAlg.HeaderParam(), err)
	}

	return nil
}

// CryptoPublicKey returns the internal object which can be used as a
// crypto.PublicKey for use with other standard library operations. The type
// is either *rsa.PublicKey or *ecdsa.PublicKey
func (k *rsaPublicKey) CryptoPublicKey() crypto.PublicKey {
	return k.PublicKey
}

func (k *rsaPublicKey) toMap() map[string]interface{} {
	jwk := make(map[string]interface{})
	for k, v := range k.extended {
		jwk[k] = v
	}
	jwk["kty"] = k.KeyType()
	jwk["kid"] = k.KeyID()
	jwk["n"] = joseBase64UrlEncode(k.N.Bytes())
	jwk["e"] = joseBase64UrlEncode(serializeRSAPublicExponentParam(k.E))

	return jwk
}

// MarshalJSON serializes this Public Key using the JWK JSON serialization format for
// RSA keys.
func (k *rsaPublicKey) MarshalJSON() (data []byte, err error) {
	return json.Marshal(k.toMap())
}

// PEMBlock serializes this Public Key to DER-encoded PKIX format.
func (k *rsaPublicKey) PEMBlock() (*pem.Block, error) {
	derBytes, err := x509.MarshalPKIXPublicKey(k.PublicKey)
	if err != nil {
		return nil, fmt.Errorf("unable to serialize RSA PublicKey to DER-encoded PKIX format: %s", err)
	}
	k.extended["kid"] = k.KeyID() // For display purposes.
	return createPemBlock("PUBLIC KEY", derBytes, k.extended)
}

func (k *rsaPublicKey) AddExtendedField(field string, value interface{}) {
	k.extended[field] = value
}

func (k *rsaPublicKey) GetExtendedField(field string) interface{} {
	v, ok := k.extended[field]
	if !ok {
		return nil
	}
	return v
}

func rsaPublicKeyFromMap(jwk map[string]interface{}) (*rsaPublicKey, error) {
	// JWK key type (kty) has already been determined to be "RSA".
	// Need to extract 'n', 'e', and 'kid' and check for
	// consistency.

	// Get the modulus parameter N.
	nB64Url, err := stringFromMap(jwk, "n")
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Public Key modulus: %s", err)
	}

	n, err := parseRSAModulusParam(nB64Url)
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Public Key modulus: %s", err)
	}

	// Get the public exponent E.
	eB64Url, err := stringFromMap(jwk, "e")
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Public Key exponent: %s", err)
	}

	e, err := parseRSAPublicExponentParam(eB64Url)
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Public Key exponent: %s", err)
	}

	key := &rsaPublicKey{
		PublicKey: &rsa.PublicKey{N: n, E: e},
	}

	// Key ID is optional, but if it exists, it should match the key.
	_, ok := jwk["kid"]
	if ok {
		kid, err := stringFromMap(jwk, "kid")
		if err != nil {
			return nil, fmt.Errorf("JWK RSA Public Key ID: %s", err)
		}
		if kid != key.KeyID() {
			return nil, fmt.Errorf("JWK RSA Public Key ID does not match: %s", kid)
		}
	}

	if _, ok := jwk["d"]; ok {
		return nil, fmt.Errorf("JWK RSA Public Key cannot contain private exponent")
	}

	key.extended = jwk

	return key, nil
}

/*
 * RSA DSA PRIVATE KEY
 */

// rsaPrivateKey implements a JWK Private Key using RSA digital signature algorithms.
type rsaPrivateKey struct {
	rsaPublicKey
	*rsa.PrivateKey
}

func fromRSAPrivateKey(cryptoPrivateKey *rsa.PrivateKey) *rsaPrivateKey {
	return &rsaPrivateKey{
		*fromRSAPublicKey(&cryptoPrivateKey.PublicKey),
		cryptoPrivateKey,
	}
}

// PublicKey returns the Public Key data associated with this Private Key.
func (k *rsaPrivateKey) PublicKey() PublicKey {
	return &k.rsaPublicKey
}

func (k *rsaPrivateKey) String() string {
	return fmt.Sprintf("RSA Private Key <%s>", k.KeyID())
}

// Sign signs the data read from the io.Reader using a signature algorithm supported
// by the RSA private key. If the specified hashing algorithm is supported by
// this key, that hash function is used to generate the signature otherwise the
// the default hashing algorithm for this key is used. Returns the signature
// and the name of the JWK signature algorithm used, e.g., "RS256", "RS384",
// "RS512".
func (k *rsaPrivateKey) Sign(data io.Reader, hashID crypto.Hash) (signature []byte, alg string, err error) {
	// Generate a signature of the data using the internal alg.
	sigAlg := rsaPKCS1v15SignatureAlgorithmForHashID(hashID)
	hasher := sigAlg.HashID().New()

	_, err = io.Copy(hasher, data)
	if err != nil {
		return nil, "", fmt.Errorf("error reading data to sign: %s", err)
	}
	hash := hasher.Sum(nil)

	signature, err = rsa.SignPKCS1v15(rand.Reader, k.PrivateKey, sigAlg.HashID(), hash)
	if err != nil {
		return nil, "", fmt.Errorf("error producing signature: %s", err)
	}

	alg = sigAlg.HeaderParam()

	return
}

// CryptoPrivateKey returns the internal object which can be used as a
// crypto.PublicKey for use with other standard library operations. The type
// is either *rsa.PublicKey or *ecdsa.PublicKey
func (k *rsaPrivateKey) CryptoPrivateKey() crypto.PrivateKey {
	return k.PrivateKey
}

func (k *rsaPrivateKey) toMap() map[string]interface{} {
	k.Precompute() // Make sure the precomputed values are stored.
	jwk := k.rsaPublicKey.toMap()

	jwk["d"] = joseBase64UrlEncode(k.D.Bytes())
	jwk["p"] = joseBase64UrlEncode(k.Primes[0].Bytes())
	jwk["q"] = joseBase64UrlEncode(k.Primes[1].Bytes())
	jwk["dp"] = joseBase64UrlEncode(k.Precomputed.Dp.Bytes())
	jwk["dq"] = joseBase64UrlEncode(k.Precomputed.Dq.Bytes())
	jwk["qi"] = joseBase64UrlEncode(k.Precomputed.Qinv.Bytes())

	otherPrimes := k.Primes[2:]

	if len(otherPrimes) > 0 {
		otherPrimesInfo := make([]interface{}, len(otherPrimes))
		for i, r := range otherPrimes {
			otherPrimeInfo := make(map[string]string, 3)
			otherPrimeInfo["r"] = joseBase64UrlEncode(r.Bytes())
			crtVal := k.Precomputed.CRTValues[i]
			otherPrimeInfo["d"] = joseBase64UrlEncode(crtVal.Exp.Bytes())
			otherPrimeInfo["t"] = joseBase64UrlEncode(crtVal.Coeff.Bytes())
			otherPrimesInfo[i] = otherPrimeInfo
		}
		jwk["oth"] = otherPrimesInfo
	}

	return jwk
}

// MarshalJSON serializes this Private Key using the JWK JSON serialization format for
// RSA keys.
func (k *rsaPrivateKey) MarshalJSON() (data []byte, err error) {
	return json.Marshal(k.toMap())
}

// PEMBlock serializes this Private Key to DER-encoded PKIX format.
func (k *rsaPrivateKey) PEMBlock() (*pem.Block, error) {
	derBytes := x509.MarshalPKCS1PrivateKey(k.PrivateKey)
	k.extended["keyID"] = k.KeyID() // For display purposes.
	return createPemBlock("RSA PRIVATE KEY", derBytes, k.extended)
}

func rsaPrivateKeyFromMap(jwk map[string]interface{}) (*rsaPrivateKey, error) {
	// The JWA spec for RSA Private Keys (draft rfc section 5.3.2) states that
	// only the private key exponent 'd' is REQUIRED, the others are just for
	// signature/decryption optimizations and SHOULD be included when the JWK
	// is produced. We MAY choose to accept a JWK which only includes 'd', but
	// we're going to go ahead and not choose to accept it without the extra
	// fields. Only the 'oth' field will be optional (for multi-prime keys).
	privateExponent, err := parseRSAPrivateKeyParamFromMap(jwk, "d")
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Private Key exponent: %s", err)
	}
	firstPrimeFactor, err := parseRSAPrivateKeyParamFromMap(jwk, "p")
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Private Key prime factor: %s", err)
	}
	secondPrimeFactor, err := parseRSAPrivateKeyParamFromMap(jwk, "q")
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Private Key prime factor: %s", err)
	}
	firstFactorCRT, err := parseRSAPrivateKeyParamFromMap(jwk, "dp")
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Private Key CRT exponent: %s", err)
	}
	secondFactorCRT, err := parseRSAPrivateKeyParamFromMap(jwk, "dq")
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Private Key CRT exponent: %s", err)
	}
	crtCoeff, err := parseRSAPrivateKeyParamFromMap(jwk, "qi")
	if err != nil {
		return nil, fmt.Errorf("JWK RSA Private Key CRT coefficient: %s", err)
	}

	var oth interface{}
	if _, ok := jwk["oth"]; ok {
		oth = jwk["oth"]
		delete(jwk, "oth")
	}

	// JWK key type (kty) has already been determined to be "RSA".
	// Need to extract the public key information, then extract the private
	// key values.
	publicKey, err := rsaPublicKeyFromMap(jwk)
	if err != nil {
		return nil, err
	}

	privateKey := &rsa.PrivateKey{
		PublicKey: *publicKey.PublicKey,
		D:         privateExponent,
		Primes:    []*big.Int{firstPrimeFactor, secondPrimeFactor},
		Precomputed: rsa.PrecomputedValues{
			Dp:   firstFactorCRT,
			Dq:   secondFactorCRT,
			Qinv: crtCoeff,
		},
	}

	if oth != nil {
		// Should be an array of more JSON objects.
		otherPrimesInfo, ok := oth.([]interface{})
		if !ok {
			return nil, errors.New("JWK RSA Private Key: Invalid other primes info: must be an array")
		}
		numOtherPrimeFactors := len(otherPrimesInfo)
		if numOtherPrimeFactors == 0 {
			return nil, errors.New("JWK RSA Privake Key: Invalid other primes info: must be absent or non-empty")
		}
		otherPrimeFactors := make([]*big.Int, numOtherPrimeFactors)
		productOfPrimes := new(big.Int).Mul(firstPrimeFactor, secondPrimeFactor)
		crtValues := make([]rsa.CRTValue, numOtherPrimeFactors)

		for i, val := range otherPrimesInfo {
			otherPrimeinfo, ok := val.(map[string]interface{})
			if !ok {
				return nil, errors.New("JWK RSA Private Key: Invalid other prime info: must be a JSON object")
			}

			otherPrimeFactor, err := parseRSAPrivateKeyParamFromMap(otherPrimeinfo, "r")
			if err != nil {
				return nil, fmt.Errorf("JWK RSA Private Key prime factor: %s", err)
			}
			otherFactorCRT, err := parseRSAPrivateKeyParamFromMap(otherPrimeinfo, "d")
			if err != nil {
				return nil, fmt.Errorf("JWK RSA Private Key CRT exponent: %s", err)
			}
			otherCrtCoeff, err := parseRSAPrivateKeyParamFromMap(otherPrimeinfo, "t")
			if err != nil {
				return nil, fmt.Errorf("JWK RSA Private Key CRT coefficient: %s", err)
			}

			crtValue := crtValues[i]
			crtValue.Exp = otherFactorCRT
			crtValue.Coeff = otherCrtCoeff
			crtValue.R = productOfPrimes
			otherPrimeFactors[i] = otherPrimeFactor
			productOfPrimes = new(big.Int).Mul(productOfPrimes, otherPrimeFactor)
		}

		privateKey.Primes = append(privateKey.Primes, otherPrimeFactors...)
		privateKey.Precomputed.CRTValues = crtValues
	}

	key := &rsaPrivateKey{
		rsaPublicKey: *publicKey,
		PrivateKey:   privateKey,
	}

	return key, nil
}

/*
 *	Key Generation Functions.
 */

func generateRSAPrivateKey(bits int) (k *rsaPrivateKey, err error) {
	k = new(rsaPrivateKey)
	k.PrivateKey, err = rsa.GenerateKey(rand.Reader, bits)
	if err != nil {
		return nil, err
	}

	k.rsaPublicKey.PublicKey = &k.PrivateKey.PublicKey
	k.extended = make(map[string]interface{})

	return
}

// GenerateRSA2048PrivateKey generates a key pair using 2048-bit RSA.
func GenerateRSA2048PrivateKey() (PrivateKey, error) {
	k, err := generateRSAPrivateKey(2048)
	if err != nil {
		return nil, fmt.Errorf("error generating RSA 2048-bit key: %s", err)
	}

	return k, nil
}

// GenerateRSA3072PrivateKey generates a key pair using 3072-bit RSA.
func GenerateRSA3072PrivateKey() (PrivateKey, error) {
	k, err := generateRSAPrivateKey(3072)
	if err != nil {
		return nil, fmt.Errorf("error generating RSA 3072-bit key: %s", err)
	}

	return k, nil
}

// GenerateRSA4096PrivateKey generates a key pair using 4096-bit RSA.
func GenerateRSA4096PrivateKey() (PrivateKey, error) {
	k, err := generateRSAPrivateKey(4096)
	if err != nil {
		return nil, fmt.Errorf("error generating RSA 4096-bit key: %s", err)
	}

	return k, nil
}
