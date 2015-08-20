// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ldap

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"

	ber "gopkg.in/asn1-ber.v1"
)

// LDAP Application Codes
const (
	ApplicationBindRequest           = 0
	ApplicationBindResponse          = 1
	ApplicationUnbindRequest         = 2
	ApplicationSearchRequest         = 3
	ApplicationSearchResultEntry     = 4
	ApplicationSearchResultDone      = 5
	ApplicationModifyRequest         = 6
	ApplicationModifyResponse        = 7
	ApplicationAddRequest            = 8
	ApplicationAddResponse           = 9
	ApplicationDelRequest            = 10
	ApplicationDelResponse           = 11
	ApplicationModifyDNRequest       = 12
	ApplicationModifyDNResponse      = 13
	ApplicationCompareRequest        = 14
	ApplicationCompareResponse       = 15
	ApplicationAbandonRequest        = 16
	ApplicationSearchResultReference = 19
	ApplicationExtendedRequest       = 23
	ApplicationExtendedResponse      = 24
)

var ApplicationMap = map[uint8]string{
	ApplicationBindRequest:           "Bind Request",
	ApplicationBindResponse:          "Bind Response",
	ApplicationUnbindRequest:         "Unbind Request",
	ApplicationSearchRequest:         "Search Request",
	ApplicationSearchResultEntry:     "Search Result Entry",
	ApplicationSearchResultDone:      "Search Result Done",
	ApplicationModifyRequest:         "Modify Request",
	ApplicationModifyResponse:        "Modify Response",
	ApplicationAddRequest:            "Add Request",
	ApplicationAddResponse:           "Add Response",
	ApplicationDelRequest:            "Del Request",
	ApplicationDelResponse:           "Del Response",
	ApplicationModifyDNRequest:       "Modify DN Request",
	ApplicationModifyDNResponse:      "Modify DN Response",
	ApplicationCompareRequest:        "Compare Request",
	ApplicationCompareResponse:       "Compare Response",
	ApplicationAbandonRequest:        "Abandon Request",
	ApplicationSearchResultReference: "Search Result Reference",
	ApplicationExtendedRequest:       "Extended Request",
	ApplicationExtendedResponse:      "Extended Response",
}

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
}

// Ldap Behera Password Policy Draft 10 (https://tools.ietf.org/html/draft-behera-ldap-password-policy-10)
const (
	BeheraPasswordExpired             = 0
	BeheraAccountLocked               = 1
	BeheraChangeAfterReset            = 2
	BeheraPasswordModNotAllowed       = 3
	BeheraMustSupplyOldPassword       = 4
	BeheraInsufficientPasswordQuality = 5
	BeheraPasswordTooShort            = 6
	BeheraPasswordTooYoung            = 7
	BeheraPasswordInHistory           = 8
)

var BeheraPasswordPolicyErrorMap = map[int8]string{
	BeheraPasswordExpired:             "Password expired",
	BeheraAccountLocked:               "Account locked",
	BeheraChangeAfterReset:            "Password must be changed",
	BeheraPasswordModNotAllowed:       "Policy prevents password modification",
	BeheraMustSupplyOldPassword:       "Policy requires old password in order to change password",
	BeheraInsufficientPasswordQuality: "Password fails quality checks",
	BeheraPasswordTooShort:            "Password is too short for policy",
	BeheraPasswordTooYoung:            "Password has been changed too recently",
	BeheraPasswordInHistory:           "New password is in list of old passwords",
}

// Adds descriptions to an LDAP Response packet for debugging
func addLDAPDescriptions(packet *ber.Packet) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = NewError(ErrorDebugging, errors.New("ldap: cannot process packet to add descriptions"))
		}
	}()
	packet.Description = "LDAP Response"
	packet.Children[0].Description = "Message ID"

	application := uint8(packet.Children[1].Tag)
	packet.Children[1].Description = ApplicationMap[application]

	switch application {
	case ApplicationBindRequest:
		addRequestDescriptions(packet)
	case ApplicationBindResponse:
		addDefaultLDAPResponseDescriptions(packet)
	case ApplicationUnbindRequest:
		addRequestDescriptions(packet)
	case ApplicationSearchRequest:
		addRequestDescriptions(packet)
	case ApplicationSearchResultEntry:
		packet.Children[1].Children[0].Description = "Object Name"
		packet.Children[1].Children[1].Description = "Attributes"
		for _, child := range packet.Children[1].Children[1].Children {
			child.Description = "Attribute"
			child.Children[0].Description = "Attribute Name"
			child.Children[1].Description = "Attribute Values"
			for _, grandchild := range child.Children[1].Children {
				grandchild.Description = "Attribute Value"
			}
		}
		if len(packet.Children) == 3 {
			addControlDescriptions(packet.Children[2])
		}
	case ApplicationSearchResultDone:
		addDefaultLDAPResponseDescriptions(packet)
	case ApplicationModifyRequest:
		addRequestDescriptions(packet)
	case ApplicationModifyResponse:
	case ApplicationAddRequest:
		addRequestDescriptions(packet)
	case ApplicationAddResponse:
	case ApplicationDelRequest:
		addRequestDescriptions(packet)
	case ApplicationDelResponse:
	case ApplicationModifyDNRequest:
		addRequestDescriptions(packet)
	case ApplicationModifyDNResponse:
	case ApplicationCompareRequest:
		addRequestDescriptions(packet)
	case ApplicationCompareResponse:
	case ApplicationAbandonRequest:
		addRequestDescriptions(packet)
	case ApplicationSearchResultReference:
	case ApplicationExtendedRequest:
		addRequestDescriptions(packet)
	case ApplicationExtendedResponse:
	}

	return nil
}

func addControlDescriptions(packet *ber.Packet) {
	packet.Description = "Controls"
	for _, child := range packet.Children {
		child.Description = "Control"
		child.Children[0].Description = "Control Type (" + ControlTypeMap[child.Children[0].Value.(string)] + ")"
		value := child.Children[1]
		if len(child.Children) == 3 {
			child.Children[1].Description = "Criticality"
			value = child.Children[2]
		}
		value.Description = "Control Value"

		switch child.Children[0].Value.(string) {
		case ControlTypePaging:
			value.Description += " (Paging)"
			if value.Value != nil {
				valueChildren := ber.DecodePacket(value.Data.Bytes())
				value.Data.Truncate(0)
				value.Value = nil
				valueChildren.Children[1].Value = valueChildren.Children[1].Data.Bytes()
				value.AppendChild(valueChildren)
			}
			value.Children[0].Description = "Real Search Control Value"
			value.Children[0].Children[0].Description = "Paging Size"
			value.Children[0].Children[1].Description = "Cookie"

		case ControlTypeBeheraPasswordPolicy:
			value.Description += " (Password Policy - Behera Draft)"
			if value.Value != nil {
				valueChildren := ber.DecodePacket(value.Data.Bytes())
				value.Data.Truncate(0)
				value.Value = nil
				value.AppendChild(valueChildren)
			}
			sequence := value.Children[0]
			for _, child := range sequence.Children {
				if child.Tag == 0 {
					//Warning
					child := child.Children[0]
					packet := ber.DecodePacket(child.Data.Bytes())
					val, ok := packet.Value.(int64)
					if ok {
						if child.Tag == 0 {
							//timeBeforeExpiration
							value.Description += " (TimeBeforeExpiration)"
							child.Value = val
						} else if child.Tag == 1 {
							//graceAuthNsRemaining
							value.Description += " (GraceAuthNsRemaining)"
							child.Value = val
						}
					}
				} else if child.Tag == 1 {
					// Error
					packet := ber.DecodePacket(child.Data.Bytes())
					val, ok := packet.Value.(int8)
					if !ok {
						val = -1
					}
					child.Description = "Error"
					child.Value = val
				}
			}
		}
	}
}

func addRequestDescriptions(packet *ber.Packet) {
	packet.Description = "LDAP Request"
	packet.Children[0].Description = "Message ID"
	packet.Children[1].Description = ApplicationMap[uint8(packet.Children[1].Tag)]
	if len(packet.Children) == 3 {
		addControlDescriptions(packet.Children[2])
	}
}

func addDefaultLDAPResponseDescriptions(packet *ber.Packet) {
	resultCode := packet.Children[1].Children[0].Value.(int64)
	packet.Children[1].Children[0].Description = "Result Code (" + LDAPResultCodeMap[uint8(resultCode)] + ")"
	packet.Children[1].Children[1].Description = "Matched DN"
	packet.Children[1].Children[2].Description = "Error Message"
	if len(packet.Children[1].Children) > 3 {
		packet.Children[1].Children[3].Description = "Referral"
	}
	if len(packet.Children) == 3 {
		addControlDescriptions(packet.Children[2])
	}
}

func DebugBinaryFile(fileName string) error {
	file, err := ioutil.ReadFile(fileName)
	if err != nil {
		return NewError(ErrorDebugging, err)
	}
	ber.PrintBytes(os.Stdout, file, "")
	packet := ber.DecodePacket(file)
	addLDAPDescriptions(packet)
	ber.PrintPacket(packet)

	return nil
}

type Error struct {
	Err        error
	ResultCode uint8
}

func (e *Error) Error() string {
	return fmt.Sprintf("LDAP Result Code %d %q: %s", e.ResultCode, LDAPResultCodeMap[e.ResultCode], e.Err.Error())
}

func NewError(resultCode uint8, err error) error {
	return &Error{ResultCode: resultCode, Err: err}
}

func getLDAPResultCode(packet *ber.Packet) (code uint8, description string) {
	if len(packet.Children) >= 2 {
		response := packet.Children[1]
		if response.ClassType == ber.ClassApplication && response.TagType == ber.TypeConstructed && len(response.Children) >= 3 {
			return uint8(response.Children[0].Value.(int64)), response.Children[2].Value.(string)
		}
	}

	return ErrorNetwork, "Invalid packet format"
}

var hex = "0123456789abcdef"

func mustEscape(c byte) bool {
	return c > 0x7f || c == '(' || c == ')' || c == '\\' || c == '*' || c == 0
}

// EscapeFilter escapes from the provided LDAP filter string the special
// characters in the set `()*\` and those out of the range 0 < c < 0x80,
// as defined in RFC4515.
func EscapeFilter(filter string) string {
	escape := 0
	for i := 0; i < len(filter); i++ {
		if mustEscape(filter[i]) {
			escape++
		}
	}
	if escape == 0 {
		return filter
	}
	buf := make([]byte, len(filter)+escape*2)
	for i, j := 0, 0; i < len(filter); i++ {
		c := filter[i]
		if mustEscape(c) {
			buf[j+0] = '\\'
			buf[j+1] = hex[c>>4]
			buf[j+2] = hex[c&0xf]
			j += 3
		} else {
			buf[j] = c
			j++
		}
	}
	return string(buf)
}
