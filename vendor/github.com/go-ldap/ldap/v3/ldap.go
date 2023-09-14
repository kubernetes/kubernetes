package ldap

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	ber "github.com/go-asn1-ber/asn1-ber"
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

// ApplicationMap contains human readable descriptions of LDAP Application Codes
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

// BeheraPasswordPolicyErrorMap contains human readable descriptions of Behera Password Policy error codes
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

var logger = log.New(os.Stderr, "", log.LstdFlags)

// Logger allows clients to override the default logger
func Logger(l *log.Logger) {
	logger = l
}

// Adds descriptions to an LDAP Response packet for debugging
func addLDAPDescriptions(packet *ber.Packet) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = NewError(ErrorDebugging, fmt.Errorf("ldap: cannot process packet to add descriptions: %s", r))
		}
	}()
	packet.Description = "LDAP Response"
	packet.Children[0].Description = "Message ID"

	application := uint8(packet.Children[1].Tag)
	packet.Children[1].Description = ApplicationMap[application]

	switch application {
	case ApplicationBindRequest:
		err = addRequestDescriptions(packet)
	case ApplicationBindResponse:
		err = addDefaultLDAPResponseDescriptions(packet)
	case ApplicationUnbindRequest:
		err = addRequestDescriptions(packet)
	case ApplicationSearchRequest:
		err = addRequestDescriptions(packet)
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
			err = addControlDescriptions(packet.Children[2])
		}
	case ApplicationSearchResultDone:
		err = addDefaultLDAPResponseDescriptions(packet)
	case ApplicationModifyRequest:
		err = addRequestDescriptions(packet)
	case ApplicationModifyResponse:
	case ApplicationAddRequest:
		err = addRequestDescriptions(packet)
	case ApplicationAddResponse:
	case ApplicationDelRequest:
		err = addRequestDescriptions(packet)
	case ApplicationDelResponse:
	case ApplicationModifyDNRequest:
		err = addRequestDescriptions(packet)
	case ApplicationModifyDNResponse:
	case ApplicationCompareRequest:
		err = addRequestDescriptions(packet)
	case ApplicationCompareResponse:
	case ApplicationAbandonRequest:
		err = addRequestDescriptions(packet)
	case ApplicationSearchResultReference:
	case ApplicationExtendedRequest:
		err = addRequestDescriptions(packet)
	case ApplicationExtendedResponse:
	}

	return err
}

func addControlDescriptions(packet *ber.Packet) error {
	packet.Description = "Controls"
	for _, child := range packet.Children {
		var value *ber.Packet
		controlType := ""
		child.Description = "Control"
		switch len(child.Children) {
		case 0:
			// at least one child is required for control type
			return fmt.Errorf("at least one child is required for control type")

		case 1:
			// just type, no criticality or value
			controlType = child.Children[0].Value.(string)
			child.Children[0].Description = "Control Type (" + ControlTypeMap[controlType] + ")"

		case 2:
			controlType = child.Children[0].Value.(string)
			child.Children[0].Description = "Control Type (" + ControlTypeMap[controlType] + ")"
			// Children[1] could be criticality or value (both are optional)
			// duck-type on whether this is a boolean
			if _, ok := child.Children[1].Value.(bool); ok {
				child.Children[1].Description = "Criticality"
			} else {
				child.Children[1].Description = "Control Value"
				value = child.Children[1]
			}

		case 3:
			// criticality and value present
			controlType = child.Children[0].Value.(string)
			child.Children[0].Description = "Control Type (" + ControlTypeMap[controlType] + ")"
			child.Children[1].Description = "Criticality"
			child.Children[2].Description = "Control Value"
			value = child.Children[2]

		default:
			// more than 3 children is invalid
			return fmt.Errorf("more than 3 children for control packet found")
		}

		if value == nil {
			continue
		}
		switch controlType {
		case ControlTypePaging:
			value.Description += " (Paging)"
			if value.Value != nil {
				valueChildren, err := ber.DecodePacketErr(value.Data.Bytes())
				if err != nil {
					return fmt.Errorf("failed to decode data bytes: %s", err)
				}
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
				valueChildren, err := ber.DecodePacketErr(value.Data.Bytes())
				if err != nil {
					return fmt.Errorf("failed to decode data bytes: %s", err)
				}
				value.Data.Truncate(0)
				value.Value = nil
				value.AppendChild(valueChildren)
			}
			sequence := value.Children[0]
			for _, child := range sequence.Children {
				if child.Tag == 0 {
					// Warning
					warningPacket := child.Children[0]
					val, err := ber.ParseInt64(warningPacket.Data.Bytes())
					if err != nil {
						return fmt.Errorf("failed to decode data bytes: %s", err)
					}
					if warningPacket.Tag == 0 {
						// timeBeforeExpiration
						value.Description += " (TimeBeforeExpiration)"
						warningPacket.Value = val
					} else if warningPacket.Tag == 1 {
						// graceAuthNsRemaining
						value.Description += " (GraceAuthNsRemaining)"
						warningPacket.Value = val
					}
				} else if child.Tag == 1 {
					// Error
					bs := child.Data.Bytes()
					if len(bs) != 1 || bs[0] > 8 {
						return fmt.Errorf("failed to decode data bytes: %s", "invalid PasswordPolicyResponse enum value")
					}
					val := int8(bs[0])
					child.Description = "Error"
					child.Value = val
				}
			}
		}
	}
	return nil
}

func addRequestDescriptions(packet *ber.Packet) error {
	packet.Description = "LDAP Request"
	packet.Children[0].Description = "Message ID"
	packet.Children[1].Description = ApplicationMap[uint8(packet.Children[1].Tag)]
	if len(packet.Children) == 3 {
		return addControlDescriptions(packet.Children[2])
	}
	return nil
}

func addDefaultLDAPResponseDescriptions(packet *ber.Packet) error {
	resultCode := uint16(LDAPResultSuccess)
	matchedDN := ""
	description := "Success"
	if err := GetLDAPError(packet); err != nil {
		resultCode = err.(*Error).ResultCode
		matchedDN = err.(*Error).MatchedDN
		description = "Error Message"
	}

	packet.Children[1].Children[0].Description = "Result Code (" + LDAPResultCodeMap[resultCode] + ")"
	packet.Children[1].Children[1].Description = "Matched DN (" + matchedDN + ")"
	packet.Children[1].Children[2].Description = description
	if len(packet.Children[1].Children) > 3 {
		packet.Children[1].Children[3].Description = "Referral"
	}
	if len(packet.Children) == 3 {
		return addControlDescriptions(packet.Children[2])
	}
	return nil
}

// DebugBinaryFile reads and prints packets from the given filename
func DebugBinaryFile(fileName string) error {
	file, err := ioutil.ReadFile(fileName)
	if err != nil {
		return NewError(ErrorDebugging, err)
	}
	ber.PrintBytes(os.Stdout, file, "")
	packet, err := ber.DecodePacketErr(file)
	if err != nil {
		return fmt.Errorf("failed to decode packet: %s", err)
	}
	if err := addLDAPDescriptions(packet); err != nil {
		return err
	}
	ber.PrintPacket(packet)

	return nil
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
