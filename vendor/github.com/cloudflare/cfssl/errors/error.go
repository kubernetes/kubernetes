package errors

import (
	"crypto/x509"
	"encoding/json"
	"fmt"
)

// Error is the error type usually returned by functions in CF SSL package.
// It contains a 4-digit error code where the most significant digit
// describes the category where the error occurred and the rest 3 digits
// describe the specific error reason.
type Error struct {
	ErrorCode int    `json:"code"`
	Message   string `json:"message"`
}

// Category is the most significant digit of the error code.
type Category int

// Reason is the last 3 digits of the error code.
type Reason int

const (
	// Success indicates no error occurred.
	Success Category = 1000 * iota // 0XXX

	// CertificateError indicates a fault in a certificate.
	CertificateError // 1XXX

	// PrivateKeyError indicates a fault in a private key.
	PrivateKeyError // 2XXX

	// IntermediatesError indicates a fault in an intermediate.
	IntermediatesError // 3XXX

	// RootError indicates a fault in a root.
	RootError // 4XXX

	// PolicyError indicates an error arising from a malformed or
	// non-existent policy, or a breach of policy.
	PolicyError // 5XXX

	// DialError indicates a network fault.
	DialError // 6XXX

	// APIClientError indicates a problem with the API client.
	APIClientError // 7XXX

	// OCSPError indicates a problem with OCSP signing
	OCSPError // 8XXX

	// CSRError indicates a problem with CSR parsing
	CSRError // 9XXX

	// CTError indicates a problem with the certificate transparency process
	CTError // 10XXX

	// CertStoreError indicates a problem with the certificate store
	CertStoreError // 11XXX
)

// None is a non-specified error.
const (
	None Reason = iota
)

// Warning code for a success
const (
	BundleExpiringBit      int = 1 << iota // 0x01
	BundleNotUbiquitousBit                 // 0x02
)

// Parsing errors
const (
	Unknown      Reason = iota // X000
	ReadFailed                 // X001
	DecodeFailed               // X002
	ParseFailed                // X003
)

// The following represent certificate non-parsing errors, and must be
// specified along with CertificateError.
const (
	// SelfSigned indicates that a certificate is self-signed and
	// cannot be used in the manner being attempted.
	SelfSigned Reason = 100 * (iota + 1) // Code 11XX

	// VerifyFailed is an X.509 verification failure. The least two
	// significant digits of 12XX is determined as the actual x509
	// error is examined.
	VerifyFailed // Code 12XX

	// BadRequest indicates that the certificate request is invalid.
	BadRequest // Code 13XX

	// MissingSerial indicates that the profile specified
	// 'ClientProvidesSerialNumbers', but the SignRequest did not include a serial
	// number.
	MissingSerial // Code 14XX
)

const (
	certificateInvalid = 10 * (iota + 1) //121X
	unknownAuthority                     //122x
)

// The following represent private-key non-parsing errors, and must be
// specified with PrivateKeyError.
const (
	// Encrypted indicates that the private key is a PKCS #8 encrypted
	// private key. At this time, CFSSL does not support decrypting
	// these keys.
	Encrypted Reason = 100 * (iota + 1) //21XX

	// NotRSAOrECC indicates that they key is not an RSA or ECC
	// private key; these are the only two private key types supported
	// at this time by CFSSL.
	NotRSAOrECC //22XX

	// KeyMismatch indicates that the private key does not match
	// the public key or certificate being presented with the key.
	KeyMismatch //23XX

	// GenerationFailed indicates that a private key could not
	// be generated.
	GenerationFailed //24XX

	// Unavailable indicates that a private key mechanism (such as
	// PKCS #11) was requested but support for that mechanism is
	// not available.
	Unavailable
)

// The following are policy-related non-parsing errors, and must be
// specified along with PolicyError.
const (
	// NoKeyUsages indicates that the profile does not permit any
	// key usages for the certificate.
	NoKeyUsages Reason = 100 * (iota + 1) // 51XX

	// InvalidPolicy indicates that policy being requested is not
	// a valid policy or does not exist.
	InvalidPolicy // 52XX

	// InvalidRequest indicates a certificate request violated the
	// constraints of the policy being applied to the request.
	InvalidRequest // 53XX

	// UnknownProfile indicates that the profile does not exist.
	UnknownProfile // 54XX
)

// The following are API client related errors, and should be
// specified with APIClientError.
const (
	// AuthenticationFailure occurs when the client is unable
	// to obtain an authentication token for the request.
	AuthenticationFailure Reason = 100 * (iota + 1)

	// JSONError wraps an encoding/json error.
	JSONError

	// IOError wraps an io/ioutil error.
	IOError

	// ClientHTTPError wraps a net/http error.
	ClientHTTPError

	// ServerRequestFailed covers any other failures from the API
	// client.
	ServerRequestFailed
)

// The following are OCSP related errors, and should be
// specified with OCSPError
const (
	// IssuerMismatch ocurs when the certificate in the OCSP signing
	// request was not issued by the CA that this responder responds for.
	IssuerMismatch Reason = 100 * (iota + 1) // 81XX

	// InvalidStatus occurs when the OCSP signing requests includes an
	// invalid value for the certificate status.
	InvalidStatus
)

// Certificate transparency related errors specified with CTError
const (
	// PrecertSubmissionFailed occurs when submitting a precertificate to
	// a log server fails
	PrecertSubmissionFailed = 100 * (iota + 1)
)

// Certificate persistence related errors specified with CertStoreError
const (
	// InsertionFailed occurs when a SQL insert query failes to complete.
	InsertionFailed = 100 * (iota + 1)
	// RecordNotFound occurs when a SQL query targeting on one unique
	// record failes to update the specified row in the table.
	RecordNotFound
)

// The error interface implementation, which formats to a JSON object string.
func (e *Error) Error() string {
	marshaled, err := json.Marshal(e)
	if err != nil {
		panic(err)
	}
	return string(marshaled)

}

// New returns an error that contains  an error code and message derived from
// the given category, reason. Currently, to avoid confusion, it is not
// allowed to create an error of category Success
func New(category Category, reason Reason) *Error {
	errorCode := int(category) + int(reason)
	var msg string
	switch category {
	case OCSPError:
		switch reason {
		case ReadFailed:
			msg = "No certificate provided"
		case IssuerMismatch:
			msg = "Certificate not issued by this issuer"
		case InvalidStatus:
			msg = "Invalid revocation status"
		}
	case CertificateError:
		switch reason {
		case Unknown:
			msg = "Unknown certificate error"
		case ReadFailed:
			msg = "Failed to read certificate"
		case DecodeFailed:
			msg = "Failed to decode certificate"
		case ParseFailed:
			msg = "Failed to parse certificate"
		case SelfSigned:
			msg = "Certificate is self signed"
		case VerifyFailed:
			msg = "Unable to verify certificate"
		case BadRequest:
			msg = "Invalid certificate request"
		case MissingSerial:
			msg = "Missing serial number in request"
		default:
			panic(fmt.Sprintf("Unsupported CFSSL error reason %d under category CertificateError.",
				reason))

		}
	case PrivateKeyError:
		switch reason {
		case Unknown:
			msg = "Unknown private key error"
		case ReadFailed:
			msg = "Failed to read private key"
		case DecodeFailed:
			msg = "Failed to decode private key"
		case ParseFailed:
			msg = "Failed to parse private key"
		case Encrypted:
			msg = "Private key is encrypted."
		case NotRSAOrECC:
			msg = "Private key algorithm is not RSA or ECC"
		case KeyMismatch:
			msg = "Private key does not match public key"
		case GenerationFailed:
			msg = "Failed to new private key"
		case Unavailable:
			msg = "Private key is unavailable"
		default:
			panic(fmt.Sprintf("Unsupported CFSSL error reason %d under category PrivateKeyError.",
				reason))
		}
	case IntermediatesError:
		switch reason {
		case Unknown:
			msg = "Unknown intermediate certificate error"
		case ReadFailed:
			msg = "Failed to read intermediate certificate"
		case DecodeFailed:
			msg = "Failed to decode intermediate certificate"
		case ParseFailed:
			msg = "Failed to parse intermediate certificate"
		default:
			panic(fmt.Sprintf("Unsupported CFSSL error reason %d under category IntermediatesError.",
				reason))
		}
	case RootError:
		switch reason {
		case Unknown:
			msg = "Unknown root certificate error"
		case ReadFailed:
			msg = "Failed to read root certificate"
		case DecodeFailed:
			msg = "Failed to decode root certificate"
		case ParseFailed:
			msg = "Failed to parse root certificate"
		default:
			panic(fmt.Sprintf("Unsupported CFSSL error reason %d under category RootError.",
				reason))
		}
	case PolicyError:
		switch reason {
		case Unknown:
			msg = "Unknown policy error"
		case NoKeyUsages:
			msg = "Invalid policy: no key usage available"
		case InvalidPolicy:
			msg = "Invalid or unknown policy"
		case InvalidRequest:
			msg = "Policy violation request"
		case UnknownProfile:
			msg = "Unknown policy profile"
		default:
			panic(fmt.Sprintf("Unsupported CFSSL error reason %d under category PolicyError.",
				reason))
		}
	case DialError:
		switch reason {
		case Unknown:
			msg = "Failed to dial remote server"
		default:
			panic(fmt.Sprintf("Unsupported CFSSL error reason %d under category DialError.",
				reason))
		}
	case APIClientError:
		switch reason {
		case AuthenticationFailure:
			msg = "API client authentication failure"
		case JSONError:
			msg = "API client JSON config error"
		case ClientHTTPError:
			msg = "API client HTTP error"
		case IOError:
			msg = "API client IO error"
		case ServerRequestFailed:
			msg = "API client error: Server request failed"
		default:
			panic(fmt.Sprintf("Unsupported CFSSL error reason %d under category APIClientError.",
				reason))
		}
	case CSRError:
		switch reason {
		case Unknown:
			msg = "CSR parsing failed due to unknown error"
		case ReadFailed:
			msg = "CSR file read failed"
		case ParseFailed:
			msg = "CSR Parsing failed"
		case DecodeFailed:
			msg = "CSR Decode failed"
		case BadRequest:
			msg = "CSR Bad request"
		default:
			panic(fmt.Sprintf("Unsupported CF-SSL error reason %d under category APIClientError.", reason))
		}
	case CTError:
		switch reason {
		case Unknown:
			msg = "Certificate transparency parsing failed due to unknown error"
		case PrecertSubmissionFailed:
			msg = "Certificate transparency precertificate submission failed"
		default:
			panic(fmt.Sprintf("Unsupported CF-SSL error reason %d under category CTError.", reason))
		}
	case CertStoreError:
		switch reason {
		case Unknown:
			msg = "Certificate store action failed due to unknown error"
		default:
			panic(fmt.Sprintf("Unsupported CF-SSL error reason %d under category CertStoreError.", reason))
		}

	default:
		panic(fmt.Sprintf("Unsupported CFSSL error type: %d.",
			category))
	}
	return &Error{ErrorCode: errorCode, Message: msg}
}

// Wrap returns an error that contains the given error and an error code derived from
// the given category, reason and the error. Currently, to avoid confusion, it is not
// allowed to create an error of category Success
func Wrap(category Category, reason Reason, err error) *Error {
	errorCode := int(category) + int(reason)
	if err == nil {
		panic("Wrap needs a supplied error to initialize.")
	}

	// do not double wrap a error
	switch err.(type) {
	case *Error:
		panic("Unable to wrap a wrapped error.")
	}

	switch category {
	case CertificateError:
		// given VerifyFailed , report the status with more detailed status code
		// for some certificate errors we care.
		if reason == VerifyFailed {
			switch errorType := err.(type) {
			case x509.CertificateInvalidError:
				errorCode += certificateInvalid + int(errorType.Reason)
			case x509.UnknownAuthorityError:
				errorCode += unknownAuthority
			}
		}
	case PrivateKeyError, IntermediatesError, RootError, PolicyError, DialError,
		APIClientError, CSRError, CTError, CertStoreError:
	// no-op, just use the error
	default:
		panic(fmt.Sprintf("Unsupported CFSSL error type: %d.",
			category))
	}

	return &Error{ErrorCode: errorCode, Message: err.Error()}

}
