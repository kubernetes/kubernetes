// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ocsp parses OCSP responses as specified in RFC 2560. OCSP responses
// are signed messages attesting to the validity of a certificate for a small
// period of time. This is used to manage revocation for X.509 certificates.
package ocsp // import "golang.org/x/crypto/ocsp"

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	_ "crypto/sha1"
	_ "crypto/sha256"
	_ "crypto/sha512"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"errors"
	"fmt"
	"math/big"
	"strconv"
	"time"
)

var idPKIXOCSPBasic = asn1.ObjectIdentifier([]int{1, 3, 6, 1, 5, 5, 7, 48, 1, 1})

// ResponseStatus contains the result of an OCSP request. See
// https://tools.ietf.org/html/rfc6960#section-2.3
type ResponseStatus int

const (
	Success       ResponseStatus = 0
	Malformed     ResponseStatus = 1
	InternalError ResponseStatus = 2
	TryLater      ResponseStatus = 3
	// Status code four is unused in OCSP. See
	// https://tools.ietf.org/html/rfc6960#section-4.2.1
	SignatureRequired ResponseStatus = 5
	Unauthorized      ResponseStatus = 6
)

func (r ResponseStatus) String() string {
	switch r {
	case Success:
		return "success"
	case Malformed:
		return "malformed"
	case InternalError:
		return "internal error"
	case TryLater:
		return "try later"
	case SignatureRequired:
		return "signature required"
	case Unauthorized:
		return "unauthorized"
	default:
		return "unknown OCSP status: " + strconv.Itoa(int(r))
	}
}

// ResponseError is an error that may be returned by ParseResponse to indicate
// that the response itself is an error, not just that its indicating that a
// certificate is revoked, unknown, etc.
type ResponseError struct {
	Status ResponseStatus
}

func (r ResponseError) Error() string {
	return "ocsp: error from server: " + r.Status.String()
}

// These are internal structures that reflect the ASN.1 structure of an OCSP
// response. See RFC 2560, section 4.2.

type certID struct {
	HashAlgorithm pkix.AlgorithmIdentifier
	NameHash      []byte
	IssuerKeyHash []byte
	SerialNumber  *big.Int
}

// https://tools.ietf.org/html/rfc2560#section-4.1.1
type ocspRequest struct {
	TBSRequest tbsRequest
}

type tbsRequest struct {
	Version       int              `asn1:"explicit,tag:0,default:0,optional"`
	RequestorName pkix.RDNSequence `asn1:"explicit,tag:1,optional"`
	RequestList   []request
}

type request struct {
	Cert certID
}

type responseASN1 struct {
	Status   asn1.Enumerated
	Response responseBytes `asn1:"explicit,tag:0,optional"`
}

type responseBytes struct {
	ResponseType asn1.ObjectIdentifier
	Response     []byte
}

type basicResponse struct {
	TBSResponseData    responseData
	SignatureAlgorithm pkix.AlgorithmIdentifier
	Signature          asn1.BitString
	Certificates       []asn1.RawValue `asn1:"explicit,tag:0,optional"`
}

type responseData struct {
	Raw            asn1.RawContent
	Version        int `asn1:"optional,default:0,explicit,tag:0"`
	RawResponderID asn1.RawValue
	ProducedAt     time.Time `asn1:"generalized"`
	Responses      []singleResponse
}

type singleResponse struct {
	CertID           certID
	Good             asn1.Flag        `asn1:"tag:0,optional"`
	Revoked          revokedInfo      `asn1:"tag:1,optional"`
	Unknown          asn1.Flag        `asn1:"tag:2,optional"`
	ThisUpdate       time.Time        `asn1:"generalized"`
	NextUpdate       time.Time        `asn1:"generalized,explicit,tag:0,optional"`
	SingleExtensions []pkix.Extension `asn1:"explicit,tag:1,optional"`
}

type revokedInfo struct {
	RevocationTime time.Time       `asn1:"generalized"`
	Reason         asn1.Enumerated `asn1:"explicit,tag:0,optional"`
}

var (
	oidSignatureMD2WithRSA      = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 2}
	oidSignatureMD5WithRSA      = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 4}
	oidSignatureSHA1WithRSA     = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 5}
	oidSignatureSHA256WithRSA   = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 11}
	oidSignatureSHA384WithRSA   = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 12}
	oidSignatureSHA512WithRSA   = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 13}
	oidSignatureDSAWithSHA1     = asn1.ObjectIdentifier{1, 2, 840, 10040, 4, 3}
	oidSignatureDSAWithSHA256   = asn1.ObjectIdentifier{2, 16, 840, 1, 101, 3, 4, 3, 2}
	oidSignatureECDSAWithSHA1   = asn1.ObjectIdentifier{1, 2, 840, 10045, 4, 1}
	oidSignatureECDSAWithSHA256 = asn1.ObjectIdentifier{1, 2, 840, 10045, 4, 3, 2}
	oidSignatureECDSAWithSHA384 = asn1.ObjectIdentifier{1, 2, 840, 10045, 4, 3, 3}
	oidSignatureECDSAWithSHA512 = asn1.ObjectIdentifier{1, 2, 840, 10045, 4, 3, 4}
)

var hashOIDs = map[crypto.Hash]asn1.ObjectIdentifier{
	crypto.SHA1:   asn1.ObjectIdentifier([]int{1, 3, 14, 3, 2, 26}),
	crypto.SHA256: asn1.ObjectIdentifier([]int{2, 16, 840, 1, 101, 3, 4, 2, 1}),
	crypto.SHA384: asn1.ObjectIdentifier([]int{2, 16, 840, 1, 101, 3, 4, 2, 2}),
	crypto.SHA512: asn1.ObjectIdentifier([]int{2, 16, 840, 1, 101, 3, 4, 2, 3}),
}

// TODO(rlb): This is also from crypto/x509, so same comment as AGL's below
var signatureAlgorithmDetails = []struct {
	algo       x509.SignatureAlgorithm
	oid        asn1.ObjectIdentifier
	pubKeyAlgo x509.PublicKeyAlgorithm
	hash       crypto.Hash
}{
	{x509.MD2WithRSA, oidSignatureMD2WithRSA, x509.RSA, crypto.Hash(0) /* no value for MD2 */},
	{x509.MD5WithRSA, oidSignatureMD5WithRSA, x509.RSA, crypto.MD5},
	{x509.SHA1WithRSA, oidSignatureSHA1WithRSA, x509.RSA, crypto.SHA1},
	{x509.SHA256WithRSA, oidSignatureSHA256WithRSA, x509.RSA, crypto.SHA256},
	{x509.SHA384WithRSA, oidSignatureSHA384WithRSA, x509.RSA, crypto.SHA384},
	{x509.SHA512WithRSA, oidSignatureSHA512WithRSA, x509.RSA, crypto.SHA512},
	{x509.DSAWithSHA1, oidSignatureDSAWithSHA1, x509.DSA, crypto.SHA1},
	{x509.DSAWithSHA256, oidSignatureDSAWithSHA256, x509.DSA, crypto.SHA256},
	{x509.ECDSAWithSHA1, oidSignatureECDSAWithSHA1, x509.ECDSA, crypto.SHA1},
	{x509.ECDSAWithSHA256, oidSignatureECDSAWithSHA256, x509.ECDSA, crypto.SHA256},
	{x509.ECDSAWithSHA384, oidSignatureECDSAWithSHA384, x509.ECDSA, crypto.SHA384},
	{x509.ECDSAWithSHA512, oidSignatureECDSAWithSHA512, x509.ECDSA, crypto.SHA512},
}

// TODO(rlb): This is also from crypto/x509, so same comment as AGL's below
func signingParamsForPublicKey(pub interface{}, requestedSigAlgo x509.SignatureAlgorithm) (hashFunc crypto.Hash, sigAlgo pkix.AlgorithmIdentifier, err error) {
	var pubType x509.PublicKeyAlgorithm

	switch pub := pub.(type) {
	case *rsa.PublicKey:
		pubType = x509.RSA
		hashFunc = crypto.SHA256
		sigAlgo.Algorithm = oidSignatureSHA256WithRSA
		sigAlgo.Parameters = asn1.RawValue{
			Tag: 5,
		}

	case *ecdsa.PublicKey:
		pubType = x509.ECDSA

		switch pub.Curve {
		case elliptic.P224(), elliptic.P256():
			hashFunc = crypto.SHA256
			sigAlgo.Algorithm = oidSignatureECDSAWithSHA256
		case elliptic.P384():
			hashFunc = crypto.SHA384
			sigAlgo.Algorithm = oidSignatureECDSAWithSHA384
		case elliptic.P521():
			hashFunc = crypto.SHA512
			sigAlgo.Algorithm = oidSignatureECDSAWithSHA512
		default:
			err = errors.New("x509: unknown elliptic curve")
		}

	default:
		err = errors.New("x509: only RSA and ECDSA keys supported")
	}

	if err != nil {
		return
	}

	if requestedSigAlgo == 0 {
		return
	}

	found := false
	for _, details := range signatureAlgorithmDetails {
		if details.algo == requestedSigAlgo {
			if details.pubKeyAlgo != pubType {
				err = errors.New("x509: requested SignatureAlgorithm does not match private key type")
				return
			}
			sigAlgo.Algorithm, hashFunc = details.oid, details.hash
			if hashFunc == 0 {
				err = errors.New("x509: cannot sign with hash function requested")
				return
			}
			found = true
			break
		}
	}

	if !found {
		err = errors.New("x509: unknown SignatureAlgorithm")
	}

	return
}

// TODO(agl): this is taken from crypto/x509 and so should probably be exported
// from crypto/x509 or crypto/x509/pkix.
func getSignatureAlgorithmFromOID(oid asn1.ObjectIdentifier) x509.SignatureAlgorithm {
	for _, details := range signatureAlgorithmDetails {
		if oid.Equal(details.oid) {
			return details.algo
		}
	}
	return x509.UnknownSignatureAlgorithm
}

// TODO(rlb): This is not taken from crypto/x509, but it's of the same general form.
func getHashAlgorithmFromOID(target asn1.ObjectIdentifier) crypto.Hash {
	for hash, oid := range hashOIDs {
		if oid.Equal(target) {
			return hash
		}
	}
	return crypto.Hash(0)
}

func getOIDFromHashAlgorithm(target crypto.Hash) asn1.ObjectIdentifier {
	for hash, oid := range hashOIDs {
		if hash == target {
			return oid
		}
	}
	return nil
}

// This is the exposed reflection of the internal OCSP structures.

// The status values that can be expressed in OCSP.  See RFC 6960.
const (
	// Good means that the certificate is valid.
	Good = iota
	// Revoked means that the certificate has been deliberately revoked.
	Revoked
	// Unknown means that the OCSP responder doesn't know about the certificate.
	Unknown
	// ServerFailed is unused and was never used (see
	// https://go-review.googlesource.com/#/c/18944). ParseResponse will
	// return a ResponseError when an error response is parsed.
	ServerFailed
)

// The enumerated reasons for revoking a certificate.  See RFC 5280.
const (
	Unspecified          = iota
	KeyCompromise        = iota
	CACompromise         = iota
	AffiliationChanged   = iota
	Superseded           = iota
	CessationOfOperation = iota
	CertificateHold      = iota
	_                    = iota
	RemoveFromCRL        = iota
	PrivilegeWithdrawn   = iota
	AACompromise         = iota
)

// Request represents an OCSP request. See RFC 6960.
type Request struct {
	HashAlgorithm  crypto.Hash
	IssuerNameHash []byte
	IssuerKeyHash  []byte
	SerialNumber   *big.Int
}

// Marshal marshals the OCSP request to ASN.1 DER encoded form.
func (req *Request) Marshal() ([]byte, error) {
	hashAlg := getOIDFromHashAlgorithm(req.HashAlgorithm)
	if hashAlg == nil {
		return nil, errors.New("Unknown hash algorithm")
	}
	return asn1.Marshal(ocspRequest{
		tbsRequest{
			Version: 0,
			RequestList: []request{
				{
					Cert: certID{
						pkix.AlgorithmIdentifier{
							Algorithm:  hashAlg,
							Parameters: asn1.RawValue{Tag: 5 /* ASN.1 NULL */},
						},
						req.IssuerNameHash,
						req.IssuerKeyHash,
						req.SerialNumber,
					},
				},
			},
		},
	})
}

// Response represents an OCSP response containing a single SingleResponse. See
// RFC 6960.
type Response struct {
	// Status is one of {Good, Revoked, Unknown}
	Status                                        int
	SerialNumber                                  *big.Int
	ProducedAt, ThisUpdate, NextUpdate, RevokedAt time.Time
	RevocationReason                              int
	Certificate                                   *x509.Certificate
	// TBSResponseData contains the raw bytes of the signed response. If
	// Certificate is nil then this can be used to verify Signature.
	TBSResponseData    []byte
	Signature          []byte
	SignatureAlgorithm x509.SignatureAlgorithm

	// IssuerHash is the hash used to compute the IssuerNameHash and IssuerKeyHash.
	// Valid values are crypto.SHA1, crypto.SHA256, crypto.SHA384, and crypto.SHA512.
	// If zero, the default is crypto.SHA1.
	IssuerHash crypto.Hash

	// RawResponderName optionally contains the DER-encoded subject of the
	// responder certificate. Exactly one of RawResponderName and
	// ResponderKeyHash is set.
	RawResponderName []byte
	// ResponderKeyHash optionally contains the SHA-1 hash of the
	// responder's public key. Exactly one of RawResponderName and
	// ResponderKeyHash is set.
	ResponderKeyHash []byte

	// Extensions contains raw X.509 extensions from the singleExtensions field
	// of the OCSP response. When parsing certificates, this can be used to
	// extract non-critical extensions that are not parsed by this package. When
	// marshaling OCSP responses, the Extensions field is ignored, see
	// ExtraExtensions.
	Extensions []pkix.Extension

	// ExtraExtensions contains extensions to be copied, raw, into any marshaled
	// OCSP response (in the singleExtensions field). Values override any
	// extensions that would otherwise be produced based on the other fields. The
	// ExtraExtensions field is not populated when parsing certificates, see
	// Extensions.
	ExtraExtensions []pkix.Extension
}

// These are pre-serialized error responses for the various non-success codes
// defined by OCSP. The Unauthorized code in particular can be used by an OCSP
// responder that supports only pre-signed responses as a response to requests
// for certificates with unknown status. See RFC 5019.
var (
	MalformedRequestErrorResponse = []byte{0x30, 0x03, 0x0A, 0x01, 0x01}
	InternalErrorErrorResponse    = []byte{0x30, 0x03, 0x0A, 0x01, 0x02}
	TryLaterErrorResponse         = []byte{0x30, 0x03, 0x0A, 0x01, 0x03}
	SigRequredErrorResponse       = []byte{0x30, 0x03, 0x0A, 0x01, 0x05}
	UnauthorizedErrorResponse     = []byte{0x30, 0x03, 0x0A, 0x01, 0x06}
)

// CheckSignatureFrom checks that the signature in resp is a valid signature
// from issuer. This should only be used if resp.Certificate is nil. Otherwise,
// the OCSP response contained an intermediate certificate that created the
// signature. That signature is checked by ParseResponse and only
// resp.Certificate remains to be validated.
func (resp *Response) CheckSignatureFrom(issuer *x509.Certificate) error {
	return issuer.CheckSignature(resp.SignatureAlgorithm, resp.TBSResponseData, resp.Signature)
}

// ParseError results from an invalid OCSP response.
type ParseError string

func (p ParseError) Error() string {
	return string(p)
}

// ParseRequest parses an OCSP request in DER form. It only supports
// requests for a single certificate. Signed requests are not supported.
// If a request includes a signature, it will result in a ParseError.
func ParseRequest(bytes []byte) (*Request, error) {
	var req ocspRequest
	rest, err := asn1.Unmarshal(bytes, &req)
	if err != nil {
		return nil, err
	}
	if len(rest) > 0 {
		return nil, ParseError("trailing data in OCSP request")
	}

	if len(req.TBSRequest.RequestList) == 0 {
		return nil, ParseError("OCSP request contains no request body")
	}
	innerRequest := req.TBSRequest.RequestList[0]

	hashFunc := getHashAlgorithmFromOID(innerRequest.Cert.HashAlgorithm.Algorithm)
	if hashFunc == crypto.Hash(0) {
		return nil, ParseError("OCSP request uses unknown hash function")
	}

	return &Request{
		HashAlgorithm:  hashFunc,
		IssuerNameHash: innerRequest.Cert.NameHash,
		IssuerKeyHash:  innerRequest.Cert.IssuerKeyHash,
		SerialNumber:   innerRequest.Cert.SerialNumber,
	}, nil
}

// ParseResponse parses an OCSP response in DER form. It only supports
// responses for a single certificate. If the response contains a certificate
// then the signature over the response is checked. If issuer is not nil then
// it will be used to validate the signature or embedded certificate.
//
// Invalid responses and parse failures will result in a ParseError.
// Error responses will result in a ResponseError.
func ParseResponse(bytes []byte, issuer *x509.Certificate) (*Response, error) {
	return ParseResponseForCert(bytes, nil, issuer)
}

// ParseResponseForCert parses an OCSP response in DER form and searches for a
// Response relating to cert. If such a Response is found and the OCSP response
// contains a certificate then the signature over the response is checked. If
// issuer is not nil then it will be used to validate the signature or embedded
// certificate.
//
// Invalid responses and parse failures will result in a ParseError.
// Error responses will result in a ResponseError.
func ParseResponseForCert(bytes []byte, cert, issuer *x509.Certificate) (*Response, error) {
	var resp responseASN1
	rest, err := asn1.Unmarshal(bytes, &resp)
	if err != nil {
		return nil, err
	}
	if len(rest) > 0 {
		return nil, ParseError("trailing data in OCSP response")
	}

	if status := ResponseStatus(resp.Status); status != Success {
		return nil, ResponseError{status}
	}

	if !resp.Response.ResponseType.Equal(idPKIXOCSPBasic) {
		return nil, ParseError("bad OCSP response type")
	}

	var basicResp basicResponse
	rest, err = asn1.Unmarshal(resp.Response.Response, &basicResp)
	if err != nil {
		return nil, err
	}

	if len(basicResp.Certificates) > 1 {
		return nil, ParseError("OCSP response contains bad number of certificates")
	}

	if n := len(basicResp.TBSResponseData.Responses); n == 0 || cert == nil && n > 1 {
		return nil, ParseError("OCSP response contains bad number of responses")
	}

	var singleResp singleResponse
	if cert == nil {
		singleResp = basicResp.TBSResponseData.Responses[0]
	} else {
		match := false
		for _, resp := range basicResp.TBSResponseData.Responses {
			if cert.SerialNumber.Cmp(resp.CertID.SerialNumber) == 0 {
				singleResp = resp
				match = true
				break
			}
		}
		if !match {
			return nil, ParseError("no response matching the supplied certificate")
		}
	}

	ret := &Response{
		TBSResponseData:    basicResp.TBSResponseData.Raw,
		Signature:          basicResp.Signature.RightAlign(),
		SignatureAlgorithm: getSignatureAlgorithmFromOID(basicResp.SignatureAlgorithm.Algorithm),
		Extensions:         singleResp.SingleExtensions,
		SerialNumber:       singleResp.CertID.SerialNumber,
		ProducedAt:         basicResp.TBSResponseData.ProducedAt,
		ThisUpdate:         singleResp.ThisUpdate,
		NextUpdate:         singleResp.NextUpdate,
	}

	// Handle the ResponderID CHOICE tag. ResponderID can be flattened into
	// TBSResponseData once https://go-review.googlesource.com/34503 has been
	// released.
	rawResponderID := basicResp.TBSResponseData.RawResponderID
	switch rawResponderID.Tag {
	case 1: // Name
		var rdn pkix.RDNSequence
		if rest, err := asn1.Unmarshal(rawResponderID.Bytes, &rdn); err != nil || len(rest) != 0 {
			return nil, ParseError("invalid responder name")
		}
		ret.RawResponderName = rawResponderID.Bytes
	case 2: // KeyHash
		if rest, err := asn1.Unmarshal(rawResponderID.Bytes, &ret.ResponderKeyHash); err != nil || len(rest) != 0 {
			return nil, ParseError("invalid responder key hash")
		}
	default:
		return nil, ParseError("invalid responder id tag")
	}

	if len(basicResp.Certificates) > 0 {
		ret.Certificate, err = x509.ParseCertificate(basicResp.Certificates[0].FullBytes)
		if err != nil {
			return nil, err
		}

		if err := ret.CheckSignatureFrom(ret.Certificate); err != nil {
			return nil, ParseError("bad signature on embedded certificate: " + err.Error())
		}

		if issuer != nil {
			if err := issuer.CheckSignature(ret.Certificate.SignatureAlgorithm, ret.Certificate.RawTBSCertificate, ret.Certificate.Signature); err != nil {
				return nil, ParseError("bad OCSP signature: " + err.Error())
			}
		}
	} else if issuer != nil {
		if err := ret.CheckSignatureFrom(issuer); err != nil {
			return nil, ParseError("bad OCSP signature: " + err.Error())
		}
	}

	for _, ext := range singleResp.SingleExtensions {
		if ext.Critical {
			return nil, ParseError("unsupported critical extension")
		}
	}

	for h, oid := range hashOIDs {
		if singleResp.CertID.HashAlgorithm.Algorithm.Equal(oid) {
			ret.IssuerHash = h
			break
		}
	}
	if ret.IssuerHash == 0 {
		return nil, ParseError("unsupported issuer hash algorithm")
	}

	switch {
	case bool(singleResp.Good):
		ret.Status = Good
	case bool(singleResp.Unknown):
		ret.Status = Unknown
	default:
		ret.Status = Revoked
		ret.RevokedAt = singleResp.Revoked.RevocationTime
		ret.RevocationReason = int(singleResp.Revoked.Reason)
	}

	return ret, nil
}

// RequestOptions contains options for constructing OCSP requests.
type RequestOptions struct {
	// Hash contains the hash function that should be used when
	// constructing the OCSP request. If zero, SHA-1 will be used.
	Hash crypto.Hash
}

func (opts *RequestOptions) hash() crypto.Hash {
	if opts == nil || opts.Hash == 0 {
		// SHA-1 is nearly universally used in OCSP.
		return crypto.SHA1
	}
	return opts.Hash
}

// CreateRequest returns a DER-encoded, OCSP request for the status of cert. If
// opts is nil then sensible defaults are used.
func CreateRequest(cert, issuer *x509.Certificate, opts *RequestOptions) ([]byte, error) {
	hashFunc := opts.hash()

	// OCSP seems to be the only place where these raw hash identifiers are
	// used. I took the following from
	// http://msdn.microsoft.com/en-us/library/ff635603.aspx
	_, ok := hashOIDs[hashFunc]
	if !ok {
		return nil, x509.ErrUnsupportedAlgorithm
	}

	if !hashFunc.Available() {
		return nil, x509.ErrUnsupportedAlgorithm
	}
	h := opts.hash().New()

	var publicKeyInfo struct {
		Algorithm pkix.AlgorithmIdentifier
		PublicKey asn1.BitString
	}
	if _, err := asn1.Unmarshal(issuer.RawSubjectPublicKeyInfo, &publicKeyInfo); err != nil {
		return nil, err
	}

	h.Write(publicKeyInfo.PublicKey.RightAlign())
	issuerKeyHash := h.Sum(nil)

	h.Reset()
	h.Write(issuer.RawSubject)
	issuerNameHash := h.Sum(nil)

	req := &Request{
		HashAlgorithm:  hashFunc,
		IssuerNameHash: issuerNameHash,
		IssuerKeyHash:  issuerKeyHash,
		SerialNumber:   cert.SerialNumber,
	}
	return req.Marshal()
}

// CreateResponse returns a DER-encoded OCSP response with the specified contents.
// The fields in the response are populated as follows:
//
// The responder cert is used to populate the responder's name field, and the
// certificate itself is provided alongside the OCSP response signature.
//
// The issuer cert is used to puplate the IssuerNameHash and IssuerKeyHash fields.
//
// The template is used to populate the SerialNumber, RevocationStatus, RevokedAt,
// RevocationReason, ThisUpdate, and NextUpdate fields.
//
// If template.IssuerHash is not set, SHA1 will be used.
//
// The ProducedAt date is automatically set to the current date, to the nearest minute.
func CreateResponse(issuer, responderCert *x509.Certificate, template Response, priv crypto.Signer) ([]byte, error) {
	var publicKeyInfo struct {
		Algorithm pkix.AlgorithmIdentifier
		PublicKey asn1.BitString
	}
	if _, err := asn1.Unmarshal(issuer.RawSubjectPublicKeyInfo, &publicKeyInfo); err != nil {
		return nil, err
	}

	if template.IssuerHash == 0 {
		template.IssuerHash = crypto.SHA1
	}
	hashOID := getOIDFromHashAlgorithm(template.IssuerHash)
	if hashOID == nil {
		return nil, errors.New("unsupported issuer hash algorithm")
	}

	if !template.IssuerHash.Available() {
		return nil, fmt.Errorf("issuer hash algorithm %v not linked into binary", template.IssuerHash)
	}
	h := template.IssuerHash.New()
	h.Write(publicKeyInfo.PublicKey.RightAlign())
	issuerKeyHash := h.Sum(nil)

	h.Reset()
	h.Write(issuer.RawSubject)
	issuerNameHash := h.Sum(nil)

	innerResponse := singleResponse{
		CertID: certID{
			HashAlgorithm: pkix.AlgorithmIdentifier{
				Algorithm:  hashOID,
				Parameters: asn1.RawValue{Tag: 5 /* ASN.1 NULL */},
			},
			NameHash:      issuerNameHash,
			IssuerKeyHash: issuerKeyHash,
			SerialNumber:  template.SerialNumber,
		},
		ThisUpdate:       template.ThisUpdate.UTC(),
		NextUpdate:       template.NextUpdate.UTC(),
		SingleExtensions: template.ExtraExtensions,
	}

	switch template.Status {
	case Good:
		innerResponse.Good = true
	case Unknown:
		innerResponse.Unknown = true
	case Revoked:
		innerResponse.Revoked = revokedInfo{
			RevocationTime: template.RevokedAt.UTC(),
			Reason:         asn1.Enumerated(template.RevocationReason),
		}
	}

	rawResponderID := asn1.RawValue{
		Class:      2, // context-specific
		Tag:        1, // Name (explicit tag)
		IsCompound: true,
		Bytes:      responderCert.RawSubject,
	}
	tbsResponseData := responseData{
		Version:        0,
		RawResponderID: rawResponderID,
		ProducedAt:     time.Now().Truncate(time.Minute).UTC(),
		Responses:      []singleResponse{innerResponse},
	}

	tbsResponseDataDER, err := asn1.Marshal(tbsResponseData)
	if err != nil {
		return nil, err
	}

	hashFunc, signatureAlgorithm, err := signingParamsForPublicKey(priv.Public(), template.SignatureAlgorithm)
	if err != nil {
		return nil, err
	}

	responseHash := hashFunc.New()
	responseHash.Write(tbsResponseDataDER)
	signature, err := priv.Sign(rand.Reader, responseHash.Sum(nil), hashFunc)
	if err != nil {
		return nil, err
	}

	response := basicResponse{
		TBSResponseData:    tbsResponseData,
		SignatureAlgorithm: signatureAlgorithm,
		Signature: asn1.BitString{
			Bytes:     signature,
			BitLength: 8 * len(signature),
		},
	}
	if template.Certificate != nil {
		response.Certificates = []asn1.RawValue{
			asn1.RawValue{FullBytes: template.Certificate.Raw},
		}
	}
	responseDER, err := asn1.Marshal(response)
	if err != nil {
		return nil, err
	}

	return asn1.Marshal(responseASN1{
		Status: asn1.Enumerated(Success),
		Response: responseBytes{
			ResponseType: idPKIXOCSPBasic,
			Response:     responseDER,
		},
	})
}
