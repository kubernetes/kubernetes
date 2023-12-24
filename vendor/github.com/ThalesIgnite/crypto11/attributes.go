package crypto11

import (
	"errors"
	"fmt"
	"strings"

	"github.com/miekg/pkcs11"
)

// AttributeType represents a PKCS#11 CK_ATTRIBUTE value.
type AttributeType = uint

// Attribute represents a PKCS#11 CK_ATTRIBUTE type.
type Attribute = pkcs11.Attribute

//noinspection GoUnusedConst,GoDeprecation
const (
	CkaClass                  = AttributeType(0x00000000)
	CkaToken                  = AttributeType(0x00000001)
	CkaPrivate                = AttributeType(0x00000002)
	CkaLabel                  = AttributeType(0x00000003)
	CkaApplication            = AttributeType(0x00000010)
	CkaValue                  = AttributeType(0x00000011)
	CkaObjectId               = AttributeType(0x00000012)
	CkaCertificateType        = AttributeType(0x00000080)
	CkaIssuer                 = AttributeType(0x00000081)
	CkaSerialNumber           = AttributeType(0x00000082)
	CkaAcIssuer               = AttributeType(0x00000083)
	CkaOwner                  = AttributeType(0x00000084)
	CkaAttrTypes              = AttributeType(0x00000085)
	CkaTrusted                = AttributeType(0x00000086)
	CkaCertificateCategory    = AttributeType(0x00000087)
	CkaJavaMIDPSecurityDomain = AttributeType(0x00000088)
	CkaUrl                    = AttributeType(0x00000089)
	CkaHashOfSubjectPublicKey = AttributeType(0x0000008A)
	CkaHashOfIssuerPublicKey  = AttributeType(0x0000008B)
	CkaNameHashAlgorithm      = AttributeType(0x0000008C)
	CkaCheckValue             = AttributeType(0x00000090)

	CkaKeyType         = AttributeType(0x00000100)
	CkaSubject         = AttributeType(0x00000101)
	CkaId              = AttributeType(0x00000102)
	CkaSensitive       = AttributeType(0x00000103)
	CkaEncrypt         = AttributeType(0x00000104)
	CkaDecrypt         = AttributeType(0x00000105)
	CkaWrap            = AttributeType(0x00000106)
	CkaUnwrap          = AttributeType(0x00000107)
	CkaSign            = AttributeType(0x00000108)
	CkaSignRecover     = AttributeType(0x00000109)
	CkaVerify          = AttributeType(0x0000010A)
	CkaVerifyRecover   = AttributeType(0x0000010B)
	CkaDerive          = AttributeType(0x0000010C)
	CkaStartDate       = AttributeType(0x00000110)
	CkaEndDate         = AttributeType(0x00000111)
	CkaModulus         = AttributeType(0x00000120)
	CkaModulusBits     = AttributeType(0x00000121)
	CkaPublicExponent  = AttributeType(0x00000122)
	CkaPrivateExponent = AttributeType(0x00000123)
	CkaPrime1          = AttributeType(0x00000124)
	CkaPrime2          = AttributeType(0x00000125)
	CkaExponent1       = AttributeType(0x00000126)
	CkaExponent2       = AttributeType(0x00000127)
	CkaCoefficient     = AttributeType(0x00000128)
	CkaPublicKeyInfo   = AttributeType(0x00000129)
	CkaPrime           = AttributeType(0x00000130)
	CkaSubprime        = AttributeType(0x00000131)
	CkaBase            = AttributeType(0x00000132)

	CkaPrimeBits    = AttributeType(0x00000133)
	CkaSubprimeBits = AttributeType(0x00000134)
	/* (To retain backwards-compatibility) */
	CkaSubPrimeBits = CkaSubprimeBits

	CkaValueBits        = AttributeType(0x00000160)
	CkaValueLen         = AttributeType(0x00000161)
	CkaExtractable      = AttributeType(0x00000162)
	CkaLocal            = AttributeType(0x00000163)
	CkaNeverExtractable = AttributeType(0x00000164)
	CkaAlwaysSensitive  = AttributeType(0x00000165)
	CkaKeyGenMechanism  = AttributeType(0x00000166)

	CkaModifiable = AttributeType(0x00000170)
	CkaCopyable   = AttributeType(0x00000171)

	/* new for v2.40 */
	CkaDestroyable = AttributeType(0x00000172)

	/* CKA_ECDSA_PARAMS is deprecated in v2.11,
	 * CKA_EC_PARAMS is preferred. */
	CkaEcdsaParams = AttributeType(0x00000180)
	CkaEcParams    = AttributeType(0x00000180)

	CkaEcPoint = AttributeType(0x00000181)

	/* CKA_SECONDARY_AUTH, CKA_AUTH_PIN_FLAGS,
	 * are new for v2.10. Deprecated in v2.11 and onwards. */
	CkaSecondaryAuth = AttributeType(0x00000200) /* Deprecated */
	CkaAuthPinFlags  = AttributeType(0x00000201) /* Deprecated */

	CkaAlwaysAuthenticate = AttributeType(0x00000202)

	CkaWrapWithTrusted = AttributeType(0x00000210)

	ckfArrayAttribute = AttributeType(0x40000000)

	CkaWrapTemplate   = ckfArrayAttribute | AttributeType(0x00000211)
	CkaUnwrapTemplate = ckfArrayAttribute | AttributeType(0x00000212)

	CkaOtpFormat               = AttributeType(0x00000220)
	CkaOtpLength               = AttributeType(0x00000221)
	CkaOtpTimeInterval         = AttributeType(0x00000222)
	CkaOtpUserFriendlyMode     = AttributeType(0x00000223)
	CkaOtpChallengeRequirement = AttributeType(0x00000224)
	CkaOtpTimeRequirement      = AttributeType(0x00000225)
	CkaOtpCounterRequirement   = AttributeType(0x00000226)
	CkaOtpPinRequirement       = AttributeType(0x00000227)
	CkaOtpCounter              = AttributeType(0x0000022E)
	CkaOtpTime                 = AttributeType(0x0000022F)
	CkaOtpUserIdentifier       = AttributeType(0x0000022A)
	CkaOtpServiceIdentifier    = AttributeType(0x0000022B)
	CkaOtpServiceLogo          = AttributeType(0x0000022C)
	CkaOtpServiceLogoType      = AttributeType(0x0000022D)

	CkaGOSTR3410Params = AttributeType(0x00000250)
	CkaGOSTR3411Params = AttributeType(0x00000251)
	CkaGOST28147Params = AttributeType(0x00000252)

	CkaHwFeatureType = AttributeType(0x00000300)
	CkaResetOnInit   = AttributeType(0x00000301)
	CkaHasReset      = AttributeType(0x00000302)

	CkaPixelX                 = AttributeType(0x00000400)
	CkaPixelY                 = AttributeType(0x00000401)
	CkaResolution             = AttributeType(0x00000402)
	CkaCharRows               = AttributeType(0x00000403)
	CkaCharColumns            = AttributeType(0x00000404)
	CkaColor                  = AttributeType(0x00000405)
	CkaBitsPerPixel           = AttributeType(0x00000406)
	CkaCharSets               = AttributeType(0x00000480)
	CkaEncodingMethods        = AttributeType(0x00000481)
	CkaMimeTypes              = AttributeType(0x00000482)
	CkaMechanismType          = AttributeType(0x00000500)
	CkaRequiredCmsAttributes  = AttributeType(0x00000501)
	CkaDefaultCmsAttributes   = AttributeType(0x00000502)
	CkaSupportedCmsAttributes = AttributeType(0x00000503)
	CkaAllowedMechanisms      = ckfArrayAttribute | AttributeType(0x00000600)
)

// NewAttribute is a helper function that populates a new Attribute for common data types. This function will
// return an error if value is not of type bool, int, uint, string, []byte or time.Time (or is nil).
func NewAttribute(attributeType AttributeType, value interface{}) (a *Attribute, err error) {
	// catch any panics from the pkcs11.NewAttribute() call to handle the error cleanly
	defer func() {
		if r := recover(); r != nil {
			err = errors.New(fmt.Sprintf("failed creating Attribute: %v", r))
		}
	}()

	pAttr := pkcs11.NewAttribute(attributeType, value)
	return pAttr, nil
}

// CopyAttribute returns a deep copy of the given Attribute.
func CopyAttribute(a *Attribute) *Attribute {
	var value []byte
	if a.Value != nil && len(a.Value) > 0 {
		value = append([]byte(nil), a.Value...)
	}
	return &pkcs11.Attribute{
		Type:  a.Type,
		Value: value,
	}
}

// An AttributeSet groups together operations that are common for a collection of Attributes.
type AttributeSet map[AttributeType]*Attribute

// NewAttributeSet creates an empty AttributeSet.
func NewAttributeSet() AttributeSet {
	return make(AttributeSet)
}

// Set stores a new attribute in the AttributeSet. Any existing value will be overwritten. This function will return an
// error if value is not of type bool, int, uint, string, []byte or time.Time (or is nil).
func (a AttributeSet) Set(attributeType AttributeType, value interface{}) error {
	attr, err := NewAttribute(attributeType, value)
	if err != nil {
		return err
	}
	a[attributeType] = attr
	return nil
}

// cloneFrom make this AttributeSet a clone of the supplied set. Values are deep copied.
func (a AttributeSet) cloneFrom(set AttributeSet) {
	for key := range a {
		delete(a, key)
	}

	// Use Copy to do the deep cloning for us
	c := set.Copy()
	for k, v := range c {
		a[k] = v
	}
}

// AddIfNotPresent adds the attributes if the Attribute Type is not already present in the AttributeSet.
func (a AttributeSet) AddIfNotPresent(additional []*Attribute) {
	for _, additionalAttr := range additional {
		// Only add the attribute if it is not already present in the Attribute map
		if _, ok := a[additionalAttr.Type]; !ok {
			a[additionalAttr.Type] = additionalAttr
		}
	}
}

// ToSlice returns a deep copy of Attributes contained in the AttributeSet.
func (a AttributeSet) ToSlice() []*Attribute {
	var attributes []*Attribute
	for _, v := range a {
		duplicateAttr := CopyAttribute(v)
		attributes = append(attributes, duplicateAttr)
	}
	return attributes
}

// Copy returns a deep copy of the AttributeSet. This function will return an error if value is not of type
// bool, int, uint, string, []byte or time.Time (or is nil).
func (a AttributeSet) Copy() AttributeSet {
	b := NewAttributeSet()
	for _, v := range a {
		b[v.Type] = CopyAttribute(v)
	}
	return b
}

// Unset removes an attribute from the attributes set. If the set does not contain the attribute, this
// is a no-op.
func (a AttributeSet) Unset(attributeType AttributeType) {
	delete(a, attributeType)
}

func (a AttributeSet) String() string {
	result := new(strings.Builder)
	for attr, value := range a {
		_, _ = fmt.Fprintf(result, "%s: %x\n", attributeTypeString(attr), value.Value)
	}
	return result.String()
}

// NewAttributeSetWithID is a helper function that populates a new slice of Attributes with the provided ID.
// This function returns an error if the ID is an empty slice.
func NewAttributeSetWithID(id []byte) (AttributeSet, error) {
	if err := notNilBytes(id, "id"); err != nil {
		return nil, err
	}
	a := NewAttributeSet()
	_ = a.Set(CkaId, id) // error not possible for []byte
	return a, nil
}

// NewAttributeSetWithIDAndLabel is a helper function that populates a new slice of Attributes with the
// provided ID and Label. This function returns an error if either the ID or the Label is an empty slice.
func NewAttributeSetWithIDAndLabel(id, label []byte) (a AttributeSet, err error) {
	if a, err = NewAttributeSetWithID(id); err != nil {
		return nil, err
	}

	if err := notNilBytes(label, "label"); err != nil {
		return nil, err
	}

	_ = a.Set(CkaLabel, label) // error not possible with []byte
	return a, nil
}

func attributeTypeString(a AttributeType) string {
	//noinspection GoDeprecation
	switch a {
	case CkaClass:
		return "CkaClass"
	case CkaToken:
		return "CkaToken"
	case CkaPrivate:
		return "CkaPrivate"
	case CkaLabel:
		return "CkaLabel"
	case CkaApplication:
		return "CkaApplication"
	case CkaValue:
		return "CkaValue"
	case CkaObjectId:
		return "CkaObjectId"
	case CkaCertificateType:
		return "CkaCertificateType"
	case CkaIssuer:
		return "CkaIssuer"
	case CkaSerialNumber:
		return "CkaSerialNumber"
	case CkaAcIssuer:
		return "CkaAcIssuer"
	case CkaOwner:
		return "CkaOwner"
	case CkaAttrTypes:
		return "CkaAttrTypes"
	case CkaTrusted:
		return "CkaTrusted"
	case CkaCertificateCategory:
		return "CkaCertificateCategory"
	case CkaJavaMIDPSecurityDomain:
		return "CkaJavaMIDPSecurityDomain"
	case CkaUrl:
		return "CkaUrl"
	case CkaHashOfSubjectPublicKey:
		return "CkaHashOfSubjectPublicKey"
	case CkaHashOfIssuerPublicKey:
		return "CkaHashOfIssuerPublicKey"
	case CkaNameHashAlgorithm:
		return "CkaNameHashAlgorithm"
	case CkaCheckValue:
		return "CkaCheckValue"

	case CkaKeyType:
		return "CkaKeyType"
	case CkaSubject:
		return "CkaSubject"
	case CkaId:
		return "CkaId"
	case CkaSensitive:
		return "CkaSensitive"
	case CkaEncrypt:
		return "CkaEncrypt"
	case CkaDecrypt:
		return "CkaDecrypt"
	case CkaWrap:
		return "CkaWrap"
	case CkaUnwrap:
		return "CkaUnwrap"
	case CkaSign:
		return "CkaSign"
	case CkaSignRecover:
		return "CkaSignRecover"
	case CkaVerify:
		return "CkaVerify"
	case CkaVerifyRecover:
		return "CkaVerifyRecover"
	case CkaDerive:
		return "CkaDerive"
	case CkaStartDate:
		return "CkaStartDate"
	case CkaEndDate:
		return "CkaEndDate"
	case CkaModulus:
		return "CkaModulus"
	case CkaModulusBits:
		return "CkaModulusBits"
	case CkaPublicExponent:
		return "CkaPublicExponent"
	case CkaPrivateExponent:
		return "CkaPrivateExponent"
	case CkaPrime1:
		return "CkaPrime1"
	case CkaPrime2:
		return "CkaPrime2"
	case CkaExponent1:
		return "CkaExponent1"
	case CkaExponent2:
		return "CkaExponent2"
	case CkaCoefficient:
		return "CkaCoefficient"
	case CkaPublicKeyInfo:
		return "CkaPublicKeyInfo"
	case CkaPrime:
		return "CkaPrime"
	case CkaSubprime:
		return "CkaSubprime"
	case CkaBase:
		return "CkaBase"

	case CkaPrimeBits:
		return "CkaPrimeBits"
	case CkaSubprimeBits:
		return "CkaSubprimeBits"

	case CkaValueBits:
		return "CkaValueBits"
	case CkaValueLen:
		return "CkaValueLen"
	case CkaExtractable:
		return "CkaExtractable"
	case CkaLocal:
		return "CkaLocal"
	case CkaNeverExtractable:
		return "CkaNeverExtractable"
	case CkaAlwaysSensitive:
		return "CkaAlwaysSensitive"
	case CkaKeyGenMechanism:
		return "CkaKeyGenMechanism"

	case CkaModifiable:
		return "CkaModifiable"
	case CkaCopyable:
		return "CkaCopyable"

	case CkaDestroyable:
		return "CkaDestroyable"

	case CkaEcParams:
		return "CkaEcParams"

	case CkaEcPoint:
		return "CkaEcPoint"

	case CkaSecondaryAuth:
		return "CkaSecondaryAuth"
	case CkaAuthPinFlags:
		return "CkaAuthPinFlags"

	case CkaAlwaysAuthenticate:
		return "CkaAlwaysAuthenticate"

	case CkaWrapWithTrusted:
		return "CkaWrapWithTrusted"

	case ckfArrayAttribute:
		return "ckfArrayAttribute"

	case CkaWrapTemplate:
		return "CkaWrapTemplate"
	case CkaUnwrapTemplate:
		return "CkaUnwrapTemplate"

	case CkaOtpFormat:
		return "CkaOtpFormat"
	case CkaOtpLength:
		return "CkaOtpLength"
	case CkaOtpTimeInterval:
		return "CkaOtpTimeInterval"
	case CkaOtpUserFriendlyMode:
		return "CkaOtpUserFriendlyMode"
	case CkaOtpChallengeRequirement:
		return "CkaOtpChallengeRequirement"
	case CkaOtpTimeRequirement:
		return "CkaOtpTimeRequirement"
	case CkaOtpCounterRequirement:
		return "CkaOtpCounterRequirement"
	case CkaOtpPinRequirement:
		return "CkaOtpPinRequirement"
	case CkaOtpCounter:
		return "CkaOtpCounter"
	case CkaOtpTime:
		return "CkaOtpTime"
	case CkaOtpUserIdentifier:
		return "CkaOtpUserIdentifier"
	case CkaOtpServiceIdentifier:
		return "CkaOtpServiceIdentifier"
	case CkaOtpServiceLogo:
		return "CkaOtpServiceLogo"
	case CkaOtpServiceLogoType:
		return "CkaOtpServiceLogoType"

	case CkaGOSTR3410Params:
		return "CkaGOSTR3410Params"
	case CkaGOSTR3411Params:
		return "CkaGOSTR3411Params"
	case CkaGOST28147Params:
		return "CkaGOST28147Params"

	case CkaHwFeatureType:
		return "CkaHwFeatureType"
	case CkaResetOnInit:
		return "CkaResetOnInit"
	case CkaHasReset:
		return "CkaHasReset"

	case CkaPixelX:
		return "CkaPixelX"
	case CkaPixelY:
		return "CkaPixelY"
	case CkaResolution:
		return "CkaResolution"
	case CkaCharRows:
		return "CkaCharRows"
	case CkaCharColumns:
		return "CkaCharColumns"
	case CkaColor:
		return "CkaColor"
	case CkaBitsPerPixel:
		return "CkaBitsPerPixel"
	case CkaCharSets:
		return "CkaCharSets"
	case CkaEncodingMethods:
		return "CkaEncodingMethods"
	case CkaMimeTypes:
		return "CkaMimeTypes"
	case CkaMechanismType:
		return "CkaMechanismType"
	case CkaRequiredCmsAttributes:
		return "CkaRequiredCmsAttributes"
	case CkaDefaultCmsAttributes:
		return "CkaDefaultCmsAttributes"
	case CkaSupportedCmsAttributes:
		return "CkaSupportedCmsAttributes"
	case CkaAllowedMechanisms:
		return "CkaAllowedMechanisms"
	default:
		return "Unknown"
	}
}
