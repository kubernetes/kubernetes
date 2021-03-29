package ldap

import (
	"fmt"

	"gopkg.in/asn1-ber.v1"
)

// LDAP Result Codes
const (
	LDAPResultSuccess                      = 0
	LDAPResultOperationsError              = 1
	LDAPResultProtocolError                = 2
	LDAPResultTimeLimitExceeded            = 3
	LDAPResultSizeLimitExceeded            = 4
	LDAPResultCompareFalse                 = 5
	LDAPResultCompareTrue                  = 6
	LDAPResultAuthMethodNotSupported       = 7
	LDAPResultStrongAuthRequired           = 8
	LDAPResultReferral                     = 10
	LDAPResultAdminLimitExceeded           = 11
	LDAPResultUnavailableCriticalExtension = 12
	LDAPResultConfidentialityRequired      = 13
	LDAPResultSaslBindInProgress           = 14
	LDAPResultNoSuchAttribute              = 16
	LDAPResultUndefinedAttributeType       = 17
	LDAPResultInappropriateMatching        = 18
	LDAPResultConstraintViolation          = 19
	LDAPResultAttributeOrValueExists       = 20
	LDAPResultInvalidAttributeSyntax       = 21
	LDAPResultNoSuchObject                 = 32
	LDAPResultAliasProblem                 = 33
	LDAPResultInvalidDNSyntax              = 34
	LDAPResultAliasDereferencingProblem    = 36
	LDAPResultInappropriateAuthentication  = 48
	LDAPResultInvalidCredentials           = 49
	LDAPResultInsufficientAccessRights     = 50
	LDAPResultBusy                         = 51
	LDAPResultUnavailable                  = 52
	LDAPResultUnwillingToPerform           = 53
	LDAPResultLoopDetect                   = 54
	LDAPResultNamingViolation              = 64
	LDAPResultObjectClassViolation         = 65
	LDAPResultNotAllowedOnNonLeaf          = 66
	LDAPResultNotAllowedOnRDN              = 67
	LDAPResultEntryAlreadyExists           = 68
	LDAPResultObjectClassModsProhibited    = 69
	LDAPResultAffectsMultipleDSAs          = 71
	LDAPResultOther                        = 80

	ErrorNetwork            = 200
	ErrorFilterCompile      = 201
	ErrorFilterDecompile    = 202
	ErrorDebugging          = 203
	ErrorUnexpectedMessage  = 204
	ErrorUnexpectedResponse = 205
)

// LDAPResultCodeMap contains string descriptions for LDAP error codes
var LDAPResultCodeMap = map[uint8]string{
	LDAPResultSuccess:                      "Success",
	LDAPResultOperationsError:              "Operations Error",
	LDAPResultProtocolError:                "Protocol Error",
	LDAPResultTimeLimitExceeded:            "Time Limit Exceeded",
	LDAPResultSizeLimitExceeded:            "Size Limit Exceeded",
	LDAPResultCompareFalse:                 "Compare False",
	LDAPResultCompareTrue:                  "Compare True",
	LDAPResultAuthMethodNotSupported:       "Auth Method Not Supported",
	LDAPResultStrongAuthRequired:           "Strong Auth Required",
	LDAPResultReferral:                     "Referral",
	LDAPResultAdminLimitExceeded:           "Admin Limit Exceeded",
	LDAPResultUnavailableCriticalExtension: "Unavailable Critical Extension",
	LDAPResultConfidentialityRequired:      "Confidentiality Required",
	LDAPResultSaslBindInProgress:           "Sasl Bind In Progress",
	LDAPResultNoSuchAttribute:              "No Such Attribute",
	LDAPResultUndefinedAttributeType:       "Undefined Attribute Type",
	LDAPResultInappropriateMatching:        "Inappropriate Matching",
	LDAPResultConstraintViolation:          "Constraint Violation",
	LDAPResultAttributeOrValueExists:       "Attribute Or Value Exists",
	LDAPResultInvalidAttributeSyntax:       "Invalid Attribute Syntax",
	LDAPResultNoSuchObject:                 "No Such Object",
	LDAPResultAliasProblem:                 "Alias Problem",
	LDAPResultInvalidDNSyntax:              "Invalid DN Syntax",
	LDAPResultAliasDereferencingProblem:    "Alias Dereferencing Problem",
	LDAPResultInappropriateAuthentication:  "Inappropriate Authentication",
	LDAPResultInvalidCredentials:           "Invalid Credentials",
	LDAPResultInsufficientAccessRights:     "Insufficient Access Rights",
	LDAPResultBusy:                         "Busy",
	LDAPResultUnavailable:                  "Unavailable",
	LDAPResultUnwillingToPerform:           "Unwilling To Perform",
	LDAPResultLoopDetect:                   "Loop Detect",
	LDAPResultNamingViolation:              "Naming Violation",
	LDAPResultObjectClassViolation:         "Object Class Violation",
	LDAPResultNotAllowedOnNonLeaf:          "Not Allowed On Non Leaf",
	LDAPResultNotAllowedOnRDN:              "Not Allowed On RDN",
	LDAPResultEntryAlreadyExists:           "Entry Already Exists",
	LDAPResultObjectClassModsProhibited:    "Object Class Mods Prohibited",
	LDAPResultAffectsMultipleDSAs:          "Affects Multiple DSAs",
	LDAPResultOther:                        "Other",

	ErrorNetwork:            "Network Error",
	ErrorFilterCompile:      "Filter Compile Error",
	ErrorFilterDecompile:    "Filter Decompile Error",
	ErrorDebugging:          "Debugging Error",
	ErrorUnexpectedMessage:  "Unexpected Message",
	ErrorUnexpectedResponse: "Unexpected Response",
}

func getLDAPResultCode(packet *ber.Packet) (code uint8, description string) {
	if packet == nil {
		return ErrorUnexpectedResponse, "Empty packet"
	} else if len(packet.Children) >= 2 {
		response := packet.Children[1]
		if response == nil {
			return ErrorUnexpectedResponse, "Empty response in packet"
		}
		if response.ClassType == ber.ClassApplication && response.TagType == ber.TypeConstructed && len(response.Children) >= 3 {
			// Children[1].Children[2] is the diagnosticMessage which is guaranteed to exist as seen here: https://tools.ietf.org/html/rfc4511#section-4.1.9
			return uint8(response.Children[0].Value.(int64)), response.Children[2].Value.(string)
		}
	}

	return ErrorNetwork, "Invalid packet format"
}

// Error holds LDAP error information
type Error struct {
	// Err is the underlying error
	Err error
	// ResultCode is the LDAP error code
	ResultCode uint8
}

func (e *Error) Error() string {
	return fmt.Sprintf("LDAP Result Code %d %q: %s", e.ResultCode, LDAPResultCodeMap[e.ResultCode], e.Err.Error())
}

// NewError creates an LDAP error with the given code and underlying error
func NewError(resultCode uint8, err error) error {
	return &Error{ResultCode: resultCode, Err: err}
}

// IsErrorWithCode returns true if the given error is an LDAP error with the given result code
func IsErrorWithCode(err error, desiredResultCode uint8) bool {
	if err == nil {
		return false
	}

	serverError, ok := err.(*Error)
	if !ok {
		return false
	}

	return serverError.ResultCode == desiredResultCode
}
