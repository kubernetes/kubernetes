package ldap

import (
	"fmt"
	"strconv"

	ber "github.com/go-asn1-ber/asn1-ber"
	"github.com/google/uuid"
)

const (
	// ControlTypePaging - https://www.ietf.org/rfc/rfc2696.txt
	ControlTypePaging = "1.2.840.113556.1.4.319"
	// ControlTypeBeheraPasswordPolicy - https://tools.ietf.org/html/draft-behera-ldap-password-policy-10
	ControlTypeBeheraPasswordPolicy = "1.3.6.1.4.1.42.2.27.8.5.1"
	// ControlTypeVChuPasswordMustChange - https://tools.ietf.org/html/draft-vchu-ldap-pwd-policy-00
	ControlTypeVChuPasswordMustChange = "2.16.840.1.113730.3.4.4"
	// ControlTypeVChuPasswordWarning - https://tools.ietf.org/html/draft-vchu-ldap-pwd-policy-00
	ControlTypeVChuPasswordWarning = "2.16.840.1.113730.3.4.5"
	// ControlTypeManageDsaIT - https://tools.ietf.org/html/rfc3296
	ControlTypeManageDsaIT = "2.16.840.1.113730.3.4.2"
	// ControlTypeWhoAmI - https://tools.ietf.org/html/rfc4532
	ControlTypeWhoAmI = "1.3.6.1.4.1.4203.1.11.3"
	// ControlTypeSubtreeDelete - https://datatracker.ietf.org/doc/html/draft-armijo-ldap-treedelete-02
	ControlTypeSubtreeDelete = "1.2.840.113556.1.4.805"

	// ControlTypeServerSideSorting - https://www.ietf.org/rfc/rfc2891.txt
	ControlTypeServerSideSorting = "1.2.840.113556.1.4.473"
	// ControlTypeServerSideSorting - https://www.ietf.org/rfc/rfc2891.txt
	ControlTypeServerSideSortingResult = "1.2.840.113556.1.4.474"

	// ControlTypeMicrosoftNotification - https://msdn.microsoft.com/en-us/library/aa366983(v=vs.85).aspx
	ControlTypeMicrosoftNotification = "1.2.840.113556.1.4.528"
	// ControlTypeMicrosoftShowDeleted - https://msdn.microsoft.com/en-us/library/aa366989(v=vs.85).aspx
	ControlTypeMicrosoftShowDeleted = "1.2.840.113556.1.4.417"
	// ControlTypeMicrosoftServerLinkTTL - https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-adts/f4f523a8-abc0-4b3a-a471-6b2fef135481?redirectedfrom=MSDN
	ControlTypeMicrosoftServerLinkTTL = "1.2.840.113556.1.4.2309"
	// ControlTypeDirSync - Active Directory DirSync - https://msdn.microsoft.com/en-us/library/aa366978(v=vs.85).aspx
	ControlTypeDirSync = "1.2.840.113556.1.4.841"

	// ControlTypeSyncRequest - https://www.ietf.org/rfc/rfc4533.txt
	ControlTypeSyncRequest = "1.3.6.1.4.1.4203.1.9.1.1"
	// ControlTypeSyncState - https://www.ietf.org/rfc/rfc4533.txt
	ControlTypeSyncState = "1.3.6.1.4.1.4203.1.9.1.2"
	// ControlTypeSyncDone - https://www.ietf.org/rfc/rfc4533.txt
	ControlTypeSyncDone = "1.3.6.1.4.1.4203.1.9.1.3"
	// ControlTypeSyncInfo - https://www.ietf.org/rfc/rfc4533.txt
	ControlTypeSyncInfo = "1.3.6.1.4.1.4203.1.9.1.4"
)

// Flags for DirSync control
const (
	DirSyncIncrementalValues   int64 = 2147483648
	DirSyncPublicDataOnly      int64 = 8192
	DirSyncAncestorsFirstOrder int64 = 2048
	DirSyncObjectSecurity      int64 = 1
)

// ControlTypeMap maps controls to text descriptions
var ControlTypeMap = map[string]string{
	ControlTypePaging:                  "Paging",
	ControlTypeBeheraPasswordPolicy:    "Password Policy - Behera Draft",
	ControlTypeManageDsaIT:             "Manage DSA IT",
	ControlTypeSubtreeDelete:           "Subtree Delete Control",
	ControlTypeMicrosoftNotification:   "Change Notification - Microsoft",
	ControlTypeMicrosoftShowDeleted:    "Show Deleted Objects - Microsoft",
	ControlTypeMicrosoftServerLinkTTL:  "Return TTL-DNs for link values with associated expiry times - Microsoft",
	ControlTypeServerSideSorting:       "Server Side Sorting Request - LDAP Control Extension for Server Side Sorting of Search Results (RFC2891)",
	ControlTypeServerSideSortingResult: "Server Side Sorting Results - LDAP Control Extension for Server Side Sorting of Search Results (RFC2891)",
	ControlTypeDirSync:                 "DirSync",
	ControlTypeSyncRequest:             "Sync Request",
	ControlTypeSyncState:               "Sync State",
	ControlTypeSyncDone:                "Sync Done",
	ControlTypeSyncInfo:                "Sync Info",
}

// Control defines an interface controls provide to encode and describe themselves
type Control interface {
	// GetControlType returns the OID
	GetControlType() string
	// Encode returns the ber packet representation
	Encode() *ber.Packet
	// String returns a human-readable description
	String() string
}

// ControlString implements the Control interface for simple controls
type ControlString struct {
	ControlType  string
	Criticality  bool
	ControlValue string
}

// GetControlType returns the OID
func (c *ControlString) GetControlType() string {
	return c.ControlType
}

// Encode returns the ber packet representation
func (c *ControlString) Encode() *ber.Packet {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, c.ControlType, "Control Type ("+ControlTypeMap[c.ControlType]+")"))
	if c.Criticality {
		packet.AppendChild(ber.NewBoolean(ber.ClassUniversal, ber.TypePrimitive, ber.TagBoolean, c.Criticality, "Criticality"))
	}
	if c.ControlValue != "" {
		packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, string(c.ControlValue), "Control Value"))
	}
	return packet
}

// String returns a human-readable description
func (c *ControlString) String() string {
	return fmt.Sprintf("Control Type: %s (%q)  Criticality: %t  Control Value: %s", ControlTypeMap[c.ControlType], c.ControlType, c.Criticality, c.ControlValue)
}

// ControlPaging implements the paging control described in https://www.ietf.org/rfc/rfc2696.txt
type ControlPaging struct {
	// PagingSize indicates the page size
	PagingSize uint32
	// Cookie is an opaque value returned by the server to track a paging cursor
	Cookie []byte
}

// GetControlType returns the OID
func (c *ControlPaging) GetControlType() string {
	return ControlTypePaging
}

// Encode returns the ber packet representation
func (c *ControlPaging) Encode() *ber.Packet {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, ControlTypePaging, "Control Type ("+ControlTypeMap[ControlTypePaging]+")"))

	p2 := ber.Encode(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, nil, "Control Value (Paging)")
	seq := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Search Control Value")
	seq.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, int64(c.PagingSize), "Paging Size"))
	cookie := ber.Encode(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, nil, "Cookie")
	cookie.Value = c.Cookie
	cookie.Data.Write(c.Cookie)
	seq.AppendChild(cookie)
	p2.AppendChild(seq)

	packet.AppendChild(p2)
	return packet
}

// String returns a human-readable description
func (c *ControlPaging) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality: %t  PagingSize: %d  Cookie: %q",
		ControlTypeMap[ControlTypePaging],
		ControlTypePaging,
		false,
		c.PagingSize,
		c.Cookie)
}

// SetCookie stores the given cookie in the paging control
func (c *ControlPaging) SetCookie(cookie []byte) {
	c.Cookie = cookie
}

// ControlBeheraPasswordPolicy implements the control described in https://tools.ietf.org/html/draft-behera-ldap-password-policy-10
type ControlBeheraPasswordPolicy struct {
	// Expire contains the number of seconds before a password will expire
	Expire int64
	// Grace indicates the remaining number of times a user will be allowed to authenticate with an expired password
	Grace int64
	// Error indicates the error code
	Error int8
	// ErrorString is a human readable error
	ErrorString string
}

// GetControlType returns the OID
func (c *ControlBeheraPasswordPolicy) GetControlType() string {
	return ControlTypeBeheraPasswordPolicy
}

// Encode returns the ber packet representation
func (c *ControlBeheraPasswordPolicy) Encode() *ber.Packet {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, ControlTypeBeheraPasswordPolicy, "Control Type ("+ControlTypeMap[ControlTypeBeheraPasswordPolicy]+")"))

	return packet
}

// String returns a human-readable description
func (c *ControlBeheraPasswordPolicy) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality: %t  Expire: %d  Grace: %d  Error: %d, ErrorString: %s",
		ControlTypeMap[ControlTypeBeheraPasswordPolicy],
		ControlTypeBeheraPasswordPolicy,
		false,
		c.Expire,
		c.Grace,
		c.Error,
		c.ErrorString)
}

// ControlVChuPasswordMustChange implements the control described in https://tools.ietf.org/html/draft-vchu-ldap-pwd-policy-00
type ControlVChuPasswordMustChange struct {
	// MustChange indicates if the password is required to be changed
	MustChange bool
}

// GetControlType returns the OID
func (c *ControlVChuPasswordMustChange) GetControlType() string {
	return ControlTypeVChuPasswordMustChange
}

// Encode returns the ber packet representation
func (c *ControlVChuPasswordMustChange) Encode() *ber.Packet {
	return nil
}

// String returns a human-readable description
func (c *ControlVChuPasswordMustChange) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality: %t  MustChange: %v",
		ControlTypeMap[ControlTypeVChuPasswordMustChange],
		ControlTypeVChuPasswordMustChange,
		false,
		c.MustChange)
}

// ControlVChuPasswordWarning implements the control described in https://tools.ietf.org/html/draft-vchu-ldap-pwd-policy-00
type ControlVChuPasswordWarning struct {
	// Expire indicates the time in seconds until the password expires
	Expire int64
}

// GetControlType returns the OID
func (c *ControlVChuPasswordWarning) GetControlType() string {
	return ControlTypeVChuPasswordWarning
}

// Encode returns the ber packet representation
func (c *ControlVChuPasswordWarning) Encode() *ber.Packet {
	return nil
}

// String returns a human-readable description
func (c *ControlVChuPasswordWarning) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality: %t  Expire: %b",
		ControlTypeMap[ControlTypeVChuPasswordWarning],
		ControlTypeVChuPasswordWarning,
		false,
		c.Expire)
}

// ControlManageDsaIT implements the control described in https://tools.ietf.org/html/rfc3296
type ControlManageDsaIT struct {
	// Criticality indicates if this control is required
	Criticality bool
}

// GetControlType returns the OID
func (c *ControlManageDsaIT) GetControlType() string {
	return ControlTypeManageDsaIT
}

// Encode returns the ber packet representation
func (c *ControlManageDsaIT) Encode() *ber.Packet {
	// FIXME
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, ControlTypeManageDsaIT, "Control Type ("+ControlTypeMap[ControlTypeManageDsaIT]+")"))
	if c.Criticality {
		packet.AppendChild(ber.NewBoolean(ber.ClassUniversal, ber.TypePrimitive, ber.TagBoolean, c.Criticality, "Criticality"))
	}
	return packet
}

// String returns a human-readable description
func (c *ControlManageDsaIT) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality: %t",
		ControlTypeMap[ControlTypeManageDsaIT],
		ControlTypeManageDsaIT,
		c.Criticality)
}

// NewControlManageDsaIT returns a ControlManageDsaIT control
func NewControlManageDsaIT(Criticality bool) *ControlManageDsaIT {
	return &ControlManageDsaIT{Criticality: Criticality}
}

// ControlMicrosoftNotification implements the control described in https://msdn.microsoft.com/en-us/library/aa366983(v=vs.85).aspx
type ControlMicrosoftNotification struct{}

// GetControlType returns the OID
func (c *ControlMicrosoftNotification) GetControlType() string {
	return ControlTypeMicrosoftNotification
}

// Encode returns the ber packet representation
func (c *ControlMicrosoftNotification) Encode() *ber.Packet {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, ControlTypeMicrosoftNotification, "Control Type ("+ControlTypeMap[ControlTypeMicrosoftNotification]+")"))

	return packet
}

// String returns a human-readable description
func (c *ControlMicrosoftNotification) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)",
		ControlTypeMap[ControlTypeMicrosoftNotification],
		ControlTypeMicrosoftNotification)
}

// NewControlMicrosoftNotification returns a ControlMicrosoftNotification control
func NewControlMicrosoftNotification() *ControlMicrosoftNotification {
	return &ControlMicrosoftNotification{}
}

// ControlMicrosoftShowDeleted implements the control described in https://msdn.microsoft.com/en-us/library/aa366989(v=vs.85).aspx
type ControlMicrosoftShowDeleted struct{}

// GetControlType returns the OID
func (c *ControlMicrosoftShowDeleted) GetControlType() string {
	return ControlTypeMicrosoftShowDeleted
}

// Encode returns the ber packet representation
func (c *ControlMicrosoftShowDeleted) Encode() *ber.Packet {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, ControlTypeMicrosoftShowDeleted, "Control Type ("+ControlTypeMap[ControlTypeMicrosoftShowDeleted]+")"))

	return packet
}

// String returns a human-readable description
func (c *ControlMicrosoftShowDeleted) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)",
		ControlTypeMap[ControlTypeMicrosoftShowDeleted],
		ControlTypeMicrosoftShowDeleted)
}

// NewControlMicrosoftShowDeleted returns a ControlMicrosoftShowDeleted control
func NewControlMicrosoftShowDeleted() *ControlMicrosoftShowDeleted {
	return &ControlMicrosoftShowDeleted{}
}

// ControlMicrosoftServerLinkTTL implements the control described in https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-adts/f4f523a8-abc0-4b3a-a471-6b2fef135481?redirectedfrom=MSDN
type ControlMicrosoftServerLinkTTL struct{}

// GetControlType returns the OID
func (c *ControlMicrosoftServerLinkTTL) GetControlType() string {
	return ControlTypeMicrosoftServerLinkTTL
}

// Encode returns the ber packet representation
func (c *ControlMicrosoftServerLinkTTL) Encode() *ber.Packet {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, ControlTypeMicrosoftServerLinkTTL, "Control Type ("+ControlTypeMap[ControlTypeMicrosoftServerLinkTTL]+")"))

	return packet
}

// String returns a human-readable description
func (c *ControlMicrosoftServerLinkTTL) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)",
		ControlTypeMap[ControlTypeMicrosoftServerLinkTTL],
		ControlTypeMicrosoftServerLinkTTL)
}

// NewControlMicrosoftServerLinkTTL returns a ControlMicrosoftServerLinkTTL control
func NewControlMicrosoftServerLinkTTL() *ControlMicrosoftServerLinkTTL {
	return &ControlMicrosoftServerLinkTTL{}
}

// FindControl returns the first control of the given type in the list, or nil
func FindControl(controls []Control, controlType string) Control {
	for _, c := range controls {
		if c.GetControlType() == controlType {
			return c
		}
	}
	return nil
}

// DecodeControl returns a control read from the given packet, or nil if no recognized control can be made
func DecodeControl(packet *ber.Packet) (Control, error) {
	var (
		ControlType = ""
		Criticality = false
		value       *ber.Packet
	)

	switch len(packet.Children) {
	case 0:
		// at least one child is required for control type
		return nil, fmt.Errorf("at least one child is required for control type")

	case 1:
		// just type, no criticality or value
		packet.Children[0].Description = "Control Type (" + ControlTypeMap[ControlType] + ")"
		ControlType = packet.Children[0].Value.(string)

	case 2:
		packet.Children[0].Description = "Control Type (" + ControlTypeMap[ControlType] + ")"
		if packet.Children[0].Value != nil {
			ControlType = packet.Children[0].Value.(string)
		} else if packet.Children[0].Data != nil {
			ControlType = packet.Children[0].Data.String()
		} else {
			return nil, fmt.Errorf("not found where to get the control type")
		}

		// Children[1] could be criticality or value (both are optional)
		// duck-type on whether this is a boolean
		if _, ok := packet.Children[1].Value.(bool); ok {
			packet.Children[1].Description = "Criticality"
			Criticality = packet.Children[1].Value.(bool)
		} else {
			packet.Children[1].Description = "Control Value"
			value = packet.Children[1]
		}

	case 3:
		packet.Children[0].Description = "Control Type (" + ControlTypeMap[ControlType] + ")"
		ControlType = packet.Children[0].Value.(string)

		packet.Children[1].Description = "Criticality"
		Criticality = packet.Children[1].Value.(bool)

		packet.Children[2].Description = "Control Value"
		value = packet.Children[2]

	default:
		// more than 3 children is invalid
		return nil, fmt.Errorf("more than 3 children is invalid for controls")
	}

	switch ControlType {
	case ControlTypeManageDsaIT:
		return NewControlManageDsaIT(Criticality), nil
	case ControlTypePaging:
		value.Description += " (Paging)"
		c := new(ControlPaging)
		if value.Value != nil {
			valueChildren, err := ber.DecodePacketErr(value.Data.Bytes())
			if err != nil {
				return nil, fmt.Errorf("failed to decode data bytes: %s", err)
			}
			value.Data.Truncate(0)
			value.Value = nil
			value.AppendChild(valueChildren)
		}
		value = value.Children[0]
		value.Description = "Search Control Value"
		value.Children[0].Description = "Paging Size"
		value.Children[1].Description = "Cookie"
		c.PagingSize = uint32(value.Children[0].Value.(int64))
		c.Cookie = value.Children[1].Data.Bytes()
		value.Children[1].Value = c.Cookie
		return c, nil
	case ControlTypeBeheraPasswordPolicy:
		value.Description += " (Password Policy - Behera)"
		c := NewControlBeheraPasswordPolicy()
		if value.Value != nil {
			valueChildren, err := ber.DecodePacketErr(value.Data.Bytes())
			if err != nil {
				return nil, fmt.Errorf("failed to decode data bytes: %s", err)
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
					return nil, fmt.Errorf("failed to decode data bytes: %s", err)
				}
				if warningPacket.Tag == 0 {
					// timeBeforeExpiration
					c.Expire = val
					warningPacket.Value = c.Expire
				} else if warningPacket.Tag == 1 {
					// graceAuthNsRemaining
					c.Grace = val
					warningPacket.Value = c.Grace
				}
			} else if child.Tag == 1 {
				// Error
				bs := child.Data.Bytes()
				if len(bs) != 1 || bs[0] > 8 {
					return nil, fmt.Errorf("failed to decode data bytes: %s", "invalid PasswordPolicyResponse enum value")
				}
				val := int8(bs[0])
				c.Error = val
				child.Value = c.Error
				c.ErrorString = BeheraPasswordPolicyErrorMap[c.Error]
			}
		}
		return c, nil
	case ControlTypeVChuPasswordMustChange:
		c := &ControlVChuPasswordMustChange{MustChange: true}
		return c, nil
	case ControlTypeVChuPasswordWarning:
		c := &ControlVChuPasswordWarning{Expire: -1}
		expireStr := ber.DecodeString(value.Data.Bytes())

		expire, err := strconv.ParseInt(expireStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse value as int: %s", err)
		}
		c.Expire = expire
		value.Value = c.Expire

		return c, nil
	case ControlTypeMicrosoftNotification:
		return NewControlMicrosoftNotification(), nil
	case ControlTypeMicrosoftShowDeleted:
		return NewControlMicrosoftShowDeleted(), nil
	case ControlTypeMicrosoftServerLinkTTL:
		return NewControlMicrosoftServerLinkTTL(), nil
	case ControlTypeSubtreeDelete:
		return NewControlSubtreeDelete(), nil
	case ControlTypeServerSideSorting:
		return NewControlServerSideSorting(value)
	case ControlTypeServerSideSortingResult:
		return NewControlServerSideSortingResult(value)
	case ControlTypeDirSync:
		value.Description += " (DirSync)"
		return NewResponseControlDirSync(value)
	case ControlTypeSyncState:
		value.Description += " (Sync State)"
		valueChildren, err := ber.DecodePacketErr(value.Data.Bytes())
		if err != nil {
			return nil, fmt.Errorf("failed to decode data bytes: %s", err)
		}
		return NewControlSyncState(valueChildren)
	case ControlTypeSyncDone:
		value.Description += " (Sync Done)"
		valueChildren, err := ber.DecodePacketErr(value.Data.Bytes())
		if err != nil {
			return nil, fmt.Errorf("failed to decode data bytes: %s", err)
		}
		return NewControlSyncDone(valueChildren)
	case ControlTypeSyncInfo:
		value.Description += " (Sync Info)"
		valueChildren, err := ber.DecodePacketErr(value.Data.Bytes())
		if err != nil {
			return nil, fmt.Errorf("failed to decode data bytes: %s", err)
		}
		return NewControlSyncInfo(valueChildren)
	default:
		c := new(ControlString)
		c.ControlType = ControlType
		c.Criticality = Criticality
		if value != nil {
			c.ControlValue = value.Value.(string)
		}
		return c, nil
	}
}

// NewControlString returns a generic control
func NewControlString(controlType string, criticality bool, controlValue string) *ControlString {
	return &ControlString{
		ControlType:  controlType,
		Criticality:  criticality,
		ControlValue: controlValue,
	}
}

// NewControlPaging returns a paging control
func NewControlPaging(pagingSize uint32) *ControlPaging {
	return &ControlPaging{PagingSize: pagingSize}
}

// NewControlBeheraPasswordPolicy returns a ControlBeheraPasswordPolicy
func NewControlBeheraPasswordPolicy() *ControlBeheraPasswordPolicy {
	return &ControlBeheraPasswordPolicy{
		Expire: -1,
		Grace:  -1,
		Error:  -1,
	}
}

// ControlSubtreeDelete implements the subtree delete control described in
// https://datatracker.ietf.org/doc/html/draft-armijo-ldap-treedelete-02
type ControlSubtreeDelete struct{}

// GetControlType returns the OID
func (c *ControlSubtreeDelete) GetControlType() string {
	return ControlTypeSubtreeDelete
}

// NewControlSubtreeDelete returns a ControlSubtreeDelete control.
func NewControlSubtreeDelete() *ControlSubtreeDelete {
	return &ControlSubtreeDelete{}
}

// Encode returns the ber packet representation
func (c *ControlSubtreeDelete) Encode() *ber.Packet {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, ControlTypeSubtreeDelete, "Control Type ("+ControlTypeMap[ControlTypeSubtreeDelete]+")"))

	return packet
}

func (c *ControlSubtreeDelete) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)",
		ControlTypeMap[ControlTypeSubtreeDelete],
		ControlTypeSubtreeDelete)
}

func encodeControls(controls []Control) *ber.Packet {
	packet := ber.Encode(ber.ClassContext, ber.TypeConstructed, 0, nil, "Controls")
	for _, control := range controls {
		packet.AppendChild(control.Encode())
	}
	return packet
}

// ControlDirSync implements the control described in https://msdn.microsoft.com/en-us/library/aa366978(v=vs.85).aspx
type ControlDirSync struct {
	Criticality  bool
	Flags        int64
	MaxAttrCount int64
	Cookie       []byte
}

// Deprecated:  Use NewRequestControlDirSync instead
func NewControlDirSync(flags int64, maxAttrCount int64, cookie []byte) *ControlDirSync {
	return NewRequestControlDirSync(flags, maxAttrCount, cookie)
}

// NewRequestControlDirSync returns a dir sync control
func NewRequestControlDirSync(
	flags int64, maxAttrCount int64, cookie []byte,
) *ControlDirSync {
	return &ControlDirSync{
		Criticality:  true,
		Flags:        flags,
		MaxAttrCount: maxAttrCount,
		Cookie:       cookie,
	}
}

// NewResponseControlDirSync returns a dir sync control
func NewResponseControlDirSync(value *ber.Packet) (*ControlDirSync, error) {
	if value.Value != nil {
		valueChildren, err := ber.DecodePacketErr(value.Data.Bytes())
		if err != nil {
			return nil, fmt.Errorf("failed to decode data bytes: %s", err)
		}
		value.Data.Truncate(0)
		value.Value = nil
		value.AppendChild(valueChildren)
	}
	child := value.Children[0]
	if len(child.Children) != 3 { // also on initial creation, Cookie is an empty string
		return nil, fmt.Errorf("invalid number of children in dirSync control")
	}
	child.Description = "DirSync Control Value"
	child.Children[0].Description = "Flags"
	child.Children[1].Description = "MaxAttrCount"
	child.Children[2].Description = "Cookie"

	cookie := child.Children[2].Data.Bytes()
	child.Children[2].Value = cookie
	return &ControlDirSync{
		Criticality:  true,
		Flags:        child.Children[0].Value.(int64),
		MaxAttrCount: child.Children[1].Value.(int64),
		Cookie:       cookie,
	}, nil
}

// GetControlType returns the OID
func (c *ControlDirSync) GetControlType() string {
	return ControlTypeDirSync
}

// String returns a human-readable description
func (c *ControlDirSync) String() string {
	return fmt.Sprintf(
		"ControlType: %s (%q) Criticality: %t ControlValue: Flags: %d MaxAttrCount: %d",
		ControlTypeMap[ControlTypeDirSync],
		ControlTypeDirSync,
		c.Criticality,
		c.Flags,
		c.MaxAttrCount,
	)
}

// Encode returns the ber packet representation
func (c *ControlDirSync) Encode() *ber.Packet {
	cookie := ber.Encode(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "", "Cookie")
	if len(c.Cookie) != 0 {
		cookie.Value = c.Cookie
		cookie.Data.Write(c.Cookie)
	}

	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, ControlTypeDirSync, "Control Type ("+ControlTypeMap[ControlTypeDirSync]+")"))
	packet.AppendChild(ber.NewLDAPBoolean(ber.ClassUniversal, ber.TypePrimitive, ber.TagBoolean, c.Criticality, "Criticality")) // must be true always

	val := ber.Encode(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, nil, "Control Value (DirSync)")
	seq := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "DirSync Control Value")
	seq.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, int64(c.Flags), "Flags"))
	seq.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, int64(c.MaxAttrCount), "MaxAttrCount"))
	seq.AppendChild(cookie)
	val.AppendChild(seq)

	packet.AppendChild(val)
	return packet
}

// SetCookie stores the given cookie in the dirSync control
func (c *ControlDirSync) SetCookie(cookie []byte) {
	c.Cookie = cookie
}

// ControlServerSideSorting

type SortKey struct {
	Reverse       bool
	AttributeType string
	MatchingRule  string
}

type ControlServerSideSorting struct {
	SortKeys []*SortKey
}

func (c *ControlServerSideSorting) GetControlType() string {
	return ControlTypeServerSideSorting
}

func NewControlServerSideSorting(value *ber.Packet) (*ControlServerSideSorting, error) {
	sortKeys := []*SortKey{}

	val := value.Children[1].Children

	if len(val) != 1 {
		return nil, fmt.Errorf("no sequence value in packet")
	}

	sequences := val[0].Children

	for i, sequence := range sequences {
		sortKey := new(SortKey)

		if len(sequence.Children) < 2 {
			return nil, fmt.Errorf("attributeType or matchingRule is missing from sequence %d", i)
		}

		sortKey.AttributeType = sequence.Children[0].Value.(string)
		sortKey.MatchingRule = sequence.Children[1].Value.(string)

		if len(sequence.Children) == 3 {
			sortKey.Reverse = sequence.Children[2].Value.(bool)
		}

		sortKeys = append(sortKeys, sortKey)
	}

	return &ControlServerSideSorting{SortKeys: sortKeys}, nil
}

func NewControlServerSideSortingWithSortKeys(sortKeys []*SortKey) *ControlServerSideSorting {
	return &ControlServerSideSorting{SortKeys: sortKeys}
}

func (c *ControlServerSideSorting) Encode() *ber.Packet {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	control := ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, c.GetControlType(), "Control Type")

	value := ber.Encode(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, nil, "Control Value")
	seqs := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "SortKeyList")

	for _, f := range c.SortKeys {
		seq := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "")

		seq.AppendChild(
			ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, f.AttributeType, "attributeType"),
		)
		seq.AppendChild(
			ber.NewString(ber.ClassContext, ber.TypePrimitive, 0, f.MatchingRule, "orderingRule"),
		)
		if f.Reverse {
			seq.AppendChild(
				ber.NewBoolean(ber.ClassContext, ber.TypePrimitive, 1, f.Reverse, "reverseOrder"),
			)
		}

		seqs.AppendChild(seq)
	}

	value.AppendChild(seqs)

	packet.AppendChild(control)
	packet.AppendChild(value)

	return packet
}

func (c *ControlServerSideSorting) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality:%t %+v",
		"Server Side Sorting",
		c.GetControlType(),
		false,
		c.SortKeys,
	)
}

// ControlServerSideSortingResponse

const (
	ControlServerSideSortingCodeSuccess                  ControlServerSideSortingCode = 0
	ControlServerSideSortingCodeOperationsError          ControlServerSideSortingCode = 1
	ControlServerSideSortingCodeTimeLimitExceeded        ControlServerSideSortingCode = 2
	ControlServerSideSortingCodeStrongAuthRequired       ControlServerSideSortingCode = 8
	ControlServerSideSortingCodeAdminLimitExceeded       ControlServerSideSortingCode = 11
	ControlServerSideSortingCodeNoSuchAttribute          ControlServerSideSortingCode = 16
	ControlServerSideSortingCodeInappropriateMatching    ControlServerSideSortingCode = 18
	ControlServerSideSortingCodeInsufficientAccessRights ControlServerSideSortingCode = 50
	ControlServerSideSortingCodeBusy                     ControlServerSideSortingCode = 51
	ControlServerSideSortingCodeUnwillingToPerform       ControlServerSideSortingCode = 53
	ControlServerSideSortingCodeOther                    ControlServerSideSortingCode = 80
)

var ControlServerSideSortingCodes = []ControlServerSideSortingCode{
	ControlServerSideSortingCodeSuccess,
	ControlServerSideSortingCodeOperationsError,
	ControlServerSideSortingCodeTimeLimitExceeded,
	ControlServerSideSortingCodeStrongAuthRequired,
	ControlServerSideSortingCodeAdminLimitExceeded,
	ControlServerSideSortingCodeNoSuchAttribute,
	ControlServerSideSortingCodeInappropriateMatching,
	ControlServerSideSortingCodeInsufficientAccessRights,
	ControlServerSideSortingCodeBusy,
	ControlServerSideSortingCodeUnwillingToPerform,
	ControlServerSideSortingCodeOther,
}

type ControlServerSideSortingCode int64

// Valid test the code contained in the control against the ControlServerSideSortingCodes slice and return an error if the code is unknown.
func (c ControlServerSideSortingCode) Valid() error {
	for _, validRet := range ControlServerSideSortingCodes {
		if c == validRet {
			return nil
		}
	}
	return fmt.Errorf("unknown return code : %d", c)
}

func NewControlServerSideSortingResult(pkt *ber.Packet) (*ControlServerSideSortingResult, error) {
	control := new(ControlServerSideSortingResult)

	if pkt == nil || len(pkt.Children) == 0 {
		// This is currently not compliant with the ServerSideSorting RFC (see https://datatracker.ietf.org/doc/html/rfc2891#section-1.2).
		// but it's necessary because there seems to be a bug in the implementation of the popular OpenLDAP server.
		//
		// See: https://github.com/go-ldap/ldap/pull/546
		return control, nil
	}

	codeInt, err := ber.ParseInt64(pkt.Children[0].Data.Bytes())
	if err != nil {
		return nil, err
	}

	if err = ControlServerSideSortingCode(codeInt).Valid(); err != nil {
		return nil, err
	}

	return control, nil
}

type ControlServerSideSortingResult struct {
	Criticality bool

	Result ControlServerSideSortingCode

	// Not populated for now. I can't get openldap to send me this value, so I think this is specific to other directory server
	// AttributeType string
}

func (control *ControlServerSideSortingResult) GetControlType() string {
	return ControlTypeServerSideSortingResult
}

func (c *ControlServerSideSortingResult) Encode() *ber.Packet {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "SortResult sequence")
	sortResult := ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagEnumerated, int64(c.Result), "SortResult")
	packet.AppendChild(sortResult)

	return packet
}

func (c *ControlServerSideSortingResult) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q) Criticality:%t ResultCode:%+v",
		"Server Side Sorting Result",
		c.GetControlType(),
		c.Criticality,
		c.Result,
	)
}

// Mode for ControlTypeSyncRequest
type ControlSyncRequestMode int64

const (
	SyncRequestModeRefreshOnly       ControlSyncRequestMode = 1
	SyncRequestModeRefreshAndPersist ControlSyncRequestMode = 3
)

// ControlSyncRequest implements the Sync Request Control described in https://www.ietf.org/rfc/rfc4533.txt
type ControlSyncRequest struct {
	Criticality bool
	Mode        ControlSyncRequestMode
	Cookie      []byte
	ReloadHint  bool
}

func NewControlSyncRequest(
	mode ControlSyncRequestMode, cookie []byte, reloadHint bool,
) *ControlSyncRequest {
	return &ControlSyncRequest{
		Criticality: true,
		Mode:        mode,
		Cookie:      cookie,
		ReloadHint:  reloadHint,
	}
}

// GetControlType returns the OID
func (c *ControlSyncRequest) GetControlType() string {
	return ControlTypeSyncRequest
}

// Encode encodes the control
func (c *ControlSyncRequest) Encode() *ber.Packet {
	_mode := int64(c.Mode)
	mode := ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagEnumerated, _mode, "Mode")
	var cookie *ber.Packet
	if len(c.Cookie) > 0 {
		cookie = ber.Encode(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, nil, "Cookie")
		cookie.Value = c.Cookie
		cookie.Data.Write(c.Cookie)
	}
	reloadHint := ber.NewBoolean(ber.ClassUniversal, ber.TypePrimitive, ber.TagBoolean, c.ReloadHint, "Reload Hint")

	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Control")
	packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, ControlTypeSyncRequest, "Control Type ("+ControlTypeMap[ControlTypeSyncRequest]+")"))
	packet.AppendChild(ber.NewBoolean(ber.ClassUniversal, ber.TypePrimitive, ber.TagBoolean, c.Criticality, "Criticality"))

	val := ber.Encode(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, nil, "Control Value (Sync Request)")
	seq := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Sync Request Value")
	seq.AppendChild(mode)
	if cookie != nil {
		seq.AppendChild(cookie)
	}
	seq.AppendChild(reloadHint)
	val.AppendChild(seq)

	packet.AppendChild(val)
	return packet
}

// String returns a human-readable description
func (c *ControlSyncRequest) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality: %t Mode: %d Cookie: %s ReloadHint: %t",
		ControlTypeMap[ControlTypeSyncRequest],
		ControlTypeSyncRequest,
		c.Criticality,
		c.Mode,
		string(c.Cookie),
		c.ReloadHint,
	)
}

// State for ControlSyncState
type ControlSyncStateState int64

const (
	SyncStatePresent ControlSyncStateState = 0
	SyncStateAdd     ControlSyncStateState = 1
	SyncStateModify  ControlSyncStateState = 2
	SyncStateDelete  ControlSyncStateState = 3
)

// ControlSyncState implements the Sync State Control described in https://www.ietf.org/rfc/rfc4533.txt
type ControlSyncState struct {
	Criticality bool
	State       ControlSyncStateState
	EntryUUID   uuid.UUID
	Cookie      []byte
}

func NewControlSyncState(pkt *ber.Packet) (*ControlSyncState, error) {
	var (
		state     ControlSyncStateState
		entryUUID uuid.UUID
		cookie    []byte
		err       error
	)
	switch len(pkt.Children) {
	case 0, 1:
		return nil, fmt.Errorf("at least two children are required: %d", len(pkt.Children))
	case 2:
		state = ControlSyncStateState(pkt.Children[0].Value.(int64))
		entryUUID, err = uuid.FromBytes(pkt.Children[1].ByteValue)
		if err != nil {
			return nil, fmt.Errorf("failed to decode uuid: %w", err)
		}
	case 3:
		state = ControlSyncStateState(pkt.Children[0].Value.(int64))
		entryUUID, err = uuid.FromBytes(pkt.Children[1].ByteValue)
		if err != nil {
			return nil, fmt.Errorf("failed to decode uuid: %w", err)
		}
		cookie = pkt.Children[2].ByteValue
	}
	return &ControlSyncState{
		Criticality: false,
		State:       state,
		EntryUUID:   entryUUID,
		Cookie:      cookie,
	}, nil
}

// GetControlType returns the OID
func (c *ControlSyncState) GetControlType() string {
	return ControlTypeSyncState
}

// Encode encodes the control
func (c *ControlSyncState) Encode() *ber.Packet {
	return nil
}

// String returns a human-readable description
func (c *ControlSyncState) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality: %t State: %d EntryUUID: %s Cookie: %s",
		ControlTypeMap[ControlTypeSyncState],
		ControlTypeSyncState,
		c.Criticality,
		c.State,
		c.EntryUUID.String(),
		string(c.Cookie),
	)
}

// ControlSyncDone implements the Sync Done Control described in https://www.ietf.org/rfc/rfc4533.txt
type ControlSyncDone struct {
	Criticality    bool
	Cookie         []byte
	RefreshDeletes bool
}

func NewControlSyncDone(pkt *ber.Packet) (*ControlSyncDone, error) {
	var (
		cookie         []byte
		refreshDeletes bool
	)
	switch len(pkt.Children) {
	case 0:
		// have nothing to do
	case 1:
		cookie = pkt.Children[0].ByteValue
	case 2:
		cookie = pkt.Children[0].ByteValue
		refreshDeletes = pkt.Children[1].Value.(bool)
	}
	return &ControlSyncDone{
		Criticality:    false,
		Cookie:         cookie,
		RefreshDeletes: refreshDeletes,
	}, nil
}

// GetControlType returns the OID
func (c *ControlSyncDone) GetControlType() string {
	return ControlTypeSyncDone
}

// Encode encodes the control
func (c *ControlSyncDone) Encode() *ber.Packet {
	return nil
}

// String returns a human-readable description
func (c *ControlSyncDone) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality: %t Cookie: %s RefreshDeletes: %t",
		ControlTypeMap[ControlTypeSyncDone],
		ControlTypeSyncDone,
		c.Criticality,
		string(c.Cookie),
		c.RefreshDeletes,
	)
}

// Tag For ControlSyncInfo
type ControlSyncInfoValue uint64

const (
	SyncInfoNewcookie      ControlSyncInfoValue = 0
	SyncInfoRefreshDelete  ControlSyncInfoValue = 1
	SyncInfoRefreshPresent ControlSyncInfoValue = 2
	SyncInfoSyncIdSet      ControlSyncInfoValue = 3
)

// ControlSyncInfoNewCookie implements a part of syncInfoValue described in https://www.ietf.org/rfc/rfc4533.txt
type ControlSyncInfoNewCookie struct {
	Cookie []byte
}

// String returns a human-readable description
func (c *ControlSyncInfoNewCookie) String() string {
	return fmt.Sprintf(
		"NewCookie[Cookie: %s]",
		string(c.Cookie),
	)
}

// ControlSyncInfoRefreshDelete implements a part of syncInfoValue described in https://www.ietf.org/rfc/rfc4533.txt
type ControlSyncInfoRefreshDelete struct {
	Cookie      []byte
	RefreshDone bool
}

// String returns a human-readable description
func (c *ControlSyncInfoRefreshDelete) String() string {
	return fmt.Sprintf(
		"RefreshDelete[Cookie: %s RefreshDone: %t]",
		string(c.Cookie),
		c.RefreshDone,
	)
}

// ControlSyncInfoRefreshPresent implements a part of syncInfoValue described in https://www.ietf.org/rfc/rfc4533.txt
type ControlSyncInfoRefreshPresent struct {
	Cookie      []byte
	RefreshDone bool
}

// String returns a human-readable description
func (c *ControlSyncInfoRefreshPresent) String() string {
	return fmt.Sprintf(
		"RefreshPresent[Cookie: %s RefreshDone: %t]",
		string(c.Cookie),
		c.RefreshDone,
	)
}

// ControlSyncInfoSyncIdSet implements a part of syncInfoValue described in https://www.ietf.org/rfc/rfc4533.txt
type ControlSyncInfoSyncIdSet struct {
	Cookie         []byte
	RefreshDeletes bool
	SyncUUIDs      []uuid.UUID
}

// String returns a human-readable description
func (c *ControlSyncInfoSyncIdSet) String() string {
	return fmt.Sprintf(
		"SyncIdSet[Cookie: %s RefreshDeletes: %t SyncUUIDs: %v]",
		string(c.Cookie),
		c.RefreshDeletes,
		c.SyncUUIDs,
	)
}

// ControlSyncInfo implements the Sync Info Control described in https://www.ietf.org/rfc/rfc4533.txt
type ControlSyncInfo struct {
	Criticality    bool
	Value          ControlSyncInfoValue
	NewCookie      *ControlSyncInfoNewCookie
	RefreshDelete  *ControlSyncInfoRefreshDelete
	RefreshPresent *ControlSyncInfoRefreshPresent
	SyncIdSet      *ControlSyncInfoSyncIdSet
}

func NewControlSyncInfo(pkt *ber.Packet) (*ControlSyncInfo, error) {
	var (
		cookie         []byte
		refreshDone    = true
		refreshDeletes bool
		syncUUIDs      []uuid.UUID
	)
	c := &ControlSyncInfo{Criticality: false}
	switch ControlSyncInfoValue(pkt.Identifier.Tag) {
	case SyncInfoNewcookie:
		c.Value = SyncInfoNewcookie
		c.NewCookie = &ControlSyncInfoNewCookie{
			Cookie: pkt.ByteValue,
		}
	case SyncInfoRefreshDelete:
		c.Value = SyncInfoRefreshDelete
		switch len(pkt.Children) {
		case 0:
			// have nothing to do
		case 1:
			cookie = pkt.Children[0].ByteValue
		case 2:
			cookie = pkt.Children[0].ByteValue
			refreshDone = pkt.Children[1].Value.(bool)
		}
		c.RefreshDelete = &ControlSyncInfoRefreshDelete{
			Cookie:      cookie,
			RefreshDone: refreshDone,
		}
	case SyncInfoRefreshPresent:
		c.Value = SyncInfoRefreshPresent
		switch len(pkt.Children) {
		case 0:
			// have nothing to do
		case 1:
			cookie = pkt.Children[0].ByteValue
		case 2:
			cookie = pkt.Children[0].ByteValue
			refreshDone = pkt.Children[1].Value.(bool)
		}
		c.RefreshPresent = &ControlSyncInfoRefreshPresent{
			Cookie:      cookie,
			RefreshDone: refreshDone,
		}
	case SyncInfoSyncIdSet:
		c.Value = SyncInfoSyncIdSet
		switch len(pkt.Children) {
		case 0:
			// have nothing to do
		case 1:
			cookie = pkt.Children[0].ByteValue
		case 2:
			cookie = pkt.Children[0].ByteValue
			refreshDeletes = pkt.Children[1].Value.(bool)
		case 3:
			cookie = pkt.Children[0].ByteValue
			refreshDeletes = pkt.Children[1].Value.(bool)
			syncUUIDs = make([]uuid.UUID, 0, len(pkt.Children[2].Children))
			for _, child := range pkt.Children[2].Children {
				u, err := uuid.FromBytes(child.ByteValue)
				if err != nil {
					return nil, fmt.Errorf("failed to decode uuid: %w", err)
				}
				syncUUIDs = append(syncUUIDs, u)
			}
		}
		c.SyncIdSet = &ControlSyncInfoSyncIdSet{
			Cookie:         cookie,
			RefreshDeletes: refreshDeletes,
			SyncUUIDs:      syncUUIDs,
		}
	default:
		return nil, fmt.Errorf("unknown sync info value: %d", pkt.Identifier.Tag)
	}
	return c, nil
}

// GetControlType returns the OID
func (c *ControlSyncInfo) GetControlType() string {
	return ControlTypeSyncInfo
}

// Encode encodes the control
func (c *ControlSyncInfo) Encode() *ber.Packet {
	return nil
}

// String returns a human-readable description
func (c *ControlSyncInfo) String() string {
	return fmt.Sprintf(
		"Control Type: %s (%q)  Criticality: %t Value: %d %s %s %s %s",
		ControlTypeMap[ControlTypeSyncInfo],
		ControlTypeSyncInfo,
		c.Criticality,
		c.Value,
		c.NewCookie,
		c.RefreshDelete,
		c.RefreshPresent,
		c.SyncIdSet,
	)
}
