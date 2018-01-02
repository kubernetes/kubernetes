package errors

import (
	"crypto/x509"
	"encoding/json"
	"errors"
	"testing"
)

func TestNew(t *testing.T) {
	err := New(CertificateError, Unknown)
	if err == nil {
		t.Fatal("Error creation failed.")
	}
	if err.ErrorCode != int(CertificateError)+int(Unknown) {
		t.Fatal("Error code construction failed.")
	}
	if err.Message != "Unknown certificate error" {
		t.Fatal("Error message construction failed.")
	}

	code := New(OCSPError, ReadFailed).ErrorCode
	if code != 8001 {
		t.Fatal("Improper error code")
	}

	code = New(OCSPError, IssuerMismatch).ErrorCode
	if code != 8100 {
		t.Fatal("Improper error code")
	}

	code = New(OCSPError, InvalidStatus).ErrorCode
	if code != 8200 {
		t.Fatal("Improper error code")
	}

	code = New(CertificateError, Unknown).ErrorCode
	if code != 1000 {
		t.Fatal("Improper error code")
	}
	code = New(CertificateError, ReadFailed).ErrorCode
	if code != 1001 {
		t.Fatal("Improper error code")
	}
	code = New(CertificateError, DecodeFailed).ErrorCode
	if code != 1002 {
		t.Fatal("Improper error code")
	}
	code = New(CertificateError, ParseFailed).ErrorCode
	if code != 1003 {
		t.Fatal("Improper error code")
	}
	code = New(CertificateError, SelfSigned).ErrorCode
	if code != 1100 {
		t.Fatal("Improper error code")
	}
	code = New(CertificateError, VerifyFailed).ErrorCode
	if code != 1200 {
		t.Fatal("Improper error code")
	}
	code = New(CertificateError, BadRequest).ErrorCode
	if code != 1300 {
		t.Fatal("Improper error code")
	}
	code = New(CertificateError, MissingSerial).ErrorCode
	if code != 1400 {
		t.Fatal("Improper error code")
	}

	code = New(PrivateKeyError, Unknown).ErrorCode
	if code != 2000 {
		t.Fatal("Improper error code")
	}
	code = New(PrivateKeyError, ReadFailed).ErrorCode
	if code != 2001 {
		t.Fatal("Improper error code")
	}
	code = New(PrivateKeyError, DecodeFailed).ErrorCode
	if code != 2002 {
		t.Fatal("Improper error code")
	}
	code = New(PrivateKeyError, ParseFailed).ErrorCode
	if code != 2003 {
		t.Fatal("Improper error code")
	}
	code = New(PrivateKeyError, Encrypted).ErrorCode
	if code != 2100 {
		t.Fatal("Improper error code")
	}
	code = New(PrivateKeyError, NotRSAOrECC).ErrorCode
	if code != 2200 {
		t.Fatal("Improper error code")
	}
	code = New(PrivateKeyError, KeyMismatch).ErrorCode
	if code != 2300 {
		t.Fatal("Improper error code")
	}
	code = New(PrivateKeyError, GenerationFailed).ErrorCode
	if code != 2400 {
		t.Fatal("Improper error code")
	}
	code = New(PrivateKeyError, Unavailable).ErrorCode
	if code != 2500 {
		t.Fatal("Improper error code")
	}

	code = New(IntermediatesError, Unknown).ErrorCode
	if code != 3000 {
		t.Fatal("Improper error code")
	}
	code = New(IntermediatesError, ReadFailed).ErrorCode
	if code != 3001 {
		t.Fatal("Improper error code")
	}
	code = New(IntermediatesError, DecodeFailed).ErrorCode
	if code != 3002 {
		t.Fatal("Improper error code")
	}
	code = New(IntermediatesError, ParseFailed).ErrorCode
	if code != 3003 {
		t.Fatal("Improper error code")
	}

	code = New(RootError, Unknown).ErrorCode
	if code != 4000 {
		t.Fatal("Improper error code")
	}
	code = New(RootError, ReadFailed).ErrorCode
	if code != 4001 {
		t.Fatal("Improper error code")
	}
	code = New(RootError, DecodeFailed).ErrorCode
	if code != 4002 {
		t.Fatal("Improper error code")
	}
	code = New(RootError, ParseFailed).ErrorCode
	if code != 4003 {
		t.Fatal("Improper error code")
	}

	code = New(PolicyError, Unknown).ErrorCode
	if code != 5000 {
		t.Fatal("Improper error code")
	}
	code = New(PolicyError, NoKeyUsages).ErrorCode
	if code != 5100 {
		t.Fatal("Improper error code")
	}
	code = New(PolicyError, InvalidPolicy).ErrorCode
	if code != 5200 {
		t.Fatal("Improper error code")
	}
	code = New(PolicyError, InvalidRequest).ErrorCode
	if code != 5300 {
		t.Fatal("Improper error code")
	}
	code = New(PolicyError, UnknownProfile).ErrorCode
	if code != 5400 {
		t.Fatal("Improper error code")
	}

	code = New(DialError, Unknown).ErrorCode
	if code != 6000 {
		t.Fatal("Improper error code")
	}

	code = New(APIClientError, AuthenticationFailure).ErrorCode
	if code != 7100 {
		t.Fatal("Improper error code")
	}
	code = New(APIClientError, JSONError).ErrorCode
	if code != 7200 {
		t.Fatal("Improper error code")
	}
	code = New(APIClientError, ClientHTTPError).ErrorCode
	if code != 7400 {
		t.Fatal("Improper error code")
	}
	code = New(APIClientError, IOError).ErrorCode
	if code != 7300 {
		t.Fatal("Improper error code")
	}
	code = New(APIClientError, ServerRequestFailed).ErrorCode
	if code != 7500 {
		t.Fatal("Improper error code")
	}

	code = New(CSRError, Unknown).ErrorCode
	if code != 9000 {
		t.Fatal("Improper error code")
	}
	code = New(CSRError, ReadFailed).ErrorCode
	if code != 9001 {
		t.Fatal("Improper error code")
	}
	code = New(CSRError, DecodeFailed).ErrorCode
	if code != 9002 {
		t.Fatal("Improper error code")
	}
	code = New(CSRError, ParseFailed).ErrorCode
	if code != 9003 {
		t.Fatal("Improper error code")
	}
	code = New(CSRError, KeyMismatch).ErrorCode
	if code != 9300 {
		t.Fatal("Improper error code")
	}
	code = New(CSRError, BadRequest).ErrorCode
	if code != 9300 {
		t.Fatal("Improper error code")
	}

	code = New(CTError, Unknown).ErrorCode
	if code != 10000 {
		t.Fatal("Improper error code")
	}
	code = New(CTError, PrecertSubmissionFailed).ErrorCode
	if code != 10100 {
		t.Fatal("Improper error code")
	}
}

func TestWrap(t *testing.T) {
	msg := "Arbitrary error message"
	err := Wrap(CertificateError, Unknown, errors.New(msg))
	if err == nil {
		t.Fatal("Error creation failed.")
	}
	if err.ErrorCode != int(CertificateError)+int(Unknown) {
		t.Fatal("Error code construction failed.")
	}
	if err.Message != msg {
		t.Fatal("Error message construction failed.")
	}

	err = Wrap(CertificateError, VerifyFailed, x509.CertificateInvalidError{Reason: x509.Expired})
	if err == nil {
		t.Fatal("Error creation failed.")
	}
	if err.ErrorCode != int(CertificateError)+int(VerifyFailed)+certificateInvalid+int(x509.Expired) {
		t.Fatal("Error code construction failed.")
	}
	if err.Message != "x509: certificate has expired or is not yet valid" {
		t.Fatal("Error message construction failed.")
	}

	err = Wrap(CertificateError, VerifyFailed, x509.UnknownAuthorityError{})
	if err == nil {
		t.Fatal("Error creation failed.")
	}

	err = Wrap(RootError, Unknown, errors.New(msg))
	if err == nil {
		t.Fatal("Error creation failed.")
	}
	if err.ErrorCode != int(RootError)+int(Unknown) {
		t.Fatal("Error code construction failed.")
	}
	if err.Message != msg {
		t.Fatal("Error message construction failed.")
	}
}

func TestMarshal(t *testing.T) {
	msg := "Arbitrary error message"
	err := Wrap(CertificateError, Unknown, errors.New(msg))
	bytes, _ := json.Marshal(err)
	var received Error
	json.Unmarshal(bytes, &received)
	if received.ErrorCode != int(CertificateError)+int(Unknown) {
		t.Fatal("Error code construction failed.")
	}
	if received.Message != msg {
		t.Fatal("Error message construction failed.")
	}
}

func TestErrorString(t *testing.T) {
	msg := "Arbitrary error message"
	err := Wrap(CertificateError, Unknown, errors.New(msg))
	str := err.Error()
	if str != `{"code":1000,"message":"`+msg+`"}` {
		t.Fatal("Incorrect Error():", str)
	}
}

func TestHTTP(t *testing.T) {
	err := NewMethodNotAllowed("GET")
	if err == nil {
		t.Fatal("New Mathod Check failed")
	}

	err = NewBadRequest(errors.New("Bad Request"))
	if err == nil {
		t.Fatal("New Bad Request Check failed")
	}

	if err.StatusCode != 400 {
		t.Fatal("New Bad Request error code construction failed")
	}

	err = NewBadRequestString("Bad Request String")
	if err == nil {
		t.Fatal("New Bad Request String Check failed")
	}

	if err.StatusCode != 400 {
		t.Fatal("New Bad Request String error code construction failed")
	}

	err = NewBadRequestMissingParameter("Request Missing Parameter")
	if err == nil {
		t.Fatal("New Bad Request Missing Parameter Check failed")
	}

	if err.StatusCode != 400 {
		t.Fatal("New Bad Request Missing Parameter error code construction failed")
	}

	err = NewBadRequestUnwantedParameter("Unwanted Parameter Present In Request")
	if err == nil {
		t.Fatal("New Bad Request Unwanted Parameter Check failed")
	}

	if err.StatusCode != 400 {
		t.Fatal("New Bad Request Unwanted Parameter error code construction failed")
	}

}

func TestHTTPErrorString(t *testing.T) {
	method := "GET"
	err := NewMethodNotAllowed(method)
	str := err.Error()
	if str != `Method is not allowed:"`+method+`"` {
		t.Fatal("Incorrect Error():", str)
	}
}
