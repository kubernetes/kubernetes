// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/google/certificate-transparency-go/asn1"
)

// IPAddressPrefix describes an IP address prefix as an ASN.1 bit string,
// where the BitLength field holds the prefix length.
type IPAddressPrefix asn1.BitString

// IPAddressRange describes an (inclusive) IP address range.
type IPAddressRange struct {
	Min IPAddressPrefix
	Max IPAddressPrefix
}

// Most relevant values for AFI from:
// http://www.iana.org/assignments/address-family-numbers.
const (
	IPv4AddressFamilyIndicator = uint16(1)
	IPv6AddressFamilyIndicator = uint16(2)
)

// IPAddressFamilyBlocks describes a set of ranges of IP addresses.
type IPAddressFamilyBlocks struct {
	// AFI holds an address family indicator from
	// http://www.iana.org/assignments/address-family-numbers.
	AFI uint16
	// SAFI holds a subsequent address family indicator from
	// http://www.iana.org/assignments/safi-namespace.
	SAFI byte
	// InheritFromIssuer indicates that the set of addresses should
	// be taken from the issuer's certificate.
	InheritFromIssuer bool
	// AddressPrefixes holds prefixes if InheritFromIssuer is false.
	AddressPrefixes []IPAddressPrefix
	// AddressRanges holds ranges if InheritFromIssuer is false.
	AddressRanges []IPAddressRange
}

// Internal types for asn1 unmarshalling.
type ipAddressFamily struct {
	AddressFamily []byte // 2-byte AFI plus optional 1 byte SAFI
	Choice        asn1.RawValue
}

// Internally, use raw asn1.BitString rather than the IPAddressPrefix
// type alias (so that asn1.Unmarshal() decodes properly).
type ipAddressRange struct {
	Min asn1.BitString
	Max asn1.BitString
}

func parseRPKIAddrBlocks(data []byte, nfe *NonFatalErrors) []*IPAddressFamilyBlocks {
	// RFC 3779 2.2.3
	//   IPAddrBlocks        ::= SEQUENCE OF IPAddressFamily
	//
	//   IPAddressFamily     ::= SEQUENCE {    -- AFI & optional SAFI --
	//      addressFamily        OCTET STRING (SIZE (2..3)),
	//      ipAddressChoice      IPAddressChoice }
	//
	//   IPAddressChoice     ::= CHOICE {
	//      inherit              NULL, -- inherit from issuer --
	//      addressesOrRanges    SEQUENCE OF IPAddressOrRange }
	//
	//   IPAddressOrRange    ::= CHOICE {
	//      addressPrefix        IPAddress,
	//      addressRange         IPAddressRange }
	//
	//   IPAddressRange      ::= SEQUENCE {
	//      min                  IPAddress,
	//      max                  IPAddress }
	//
	//   IPAddress           ::= BIT STRING

	var addrBlocks []ipAddressFamily
	if rest, err := asn1.Unmarshal(data, &addrBlocks); err != nil {
		nfe.AddError(fmt.Errorf("failed to asn1.Unmarshal ipAddrBlocks extension: %v", err))
		return nil
	} else if len(rest) != 0 {
		nfe.AddError(errors.New("trailing data after ipAddrBlocks extension"))
		return nil
	}

	var results []*IPAddressFamilyBlocks
	for i, block := range addrBlocks {
		var fam IPAddressFamilyBlocks
		if l := len(block.AddressFamily); l < 2 || l > 3 {
			nfe.AddError(fmt.Errorf("invalid address family length (%d) for ipAddrBlock.addressFamily", l))
			continue
		}
		fam.AFI = binary.BigEndian.Uint16(block.AddressFamily[0:2])
		if len(block.AddressFamily) > 2 {
			fam.SAFI = block.AddressFamily[2]
		}
		// IPAddressChoice is an ASN.1 CHOICE where the chosen alternative is indicated by (implicit)
		// tagging of the alternatives -- here, either NULL or SEQUENCE OF.
		if bytes.Equal(block.Choice.FullBytes, asn1.NullBytes) {
			fam.InheritFromIssuer = true
			results = append(results, &fam)
			continue
		}

		var addrRanges []asn1.RawValue
		if _, err := asn1.Unmarshal(block.Choice.FullBytes, &addrRanges); err != nil {
			nfe.AddError(fmt.Errorf("failed to asn1.Unmarshal ipAddrBlocks[%d].ipAddressChoice.addressesOrRanges: %v", i, err))
			continue
		}
		for j, ar := range addrRanges {
			// Each IPAddressOrRange is a CHOICE where the alternatives have distinct (implicit)
			// tags -- here, either BIT STRING or SEQUENCE.
			switch ar.Tag {
			case asn1.TagBitString:
				// BIT STRING for single prefix IPAddress
				var val asn1.BitString
				if _, err := asn1.Unmarshal(ar.FullBytes, &val); err != nil {
					nfe.AddError(fmt.Errorf("failed to asn1.Unmarshal ipAddrBlocks[%d].ipAddressChoice.addressesOrRanges[%d].addressPrefix: %v", i, j, err))
					continue
				}
				fam.AddressPrefixes = append(fam.AddressPrefixes, IPAddressPrefix(val))

			case asn1.TagSequence:
				var val ipAddressRange
				if _, err := asn1.Unmarshal(ar.FullBytes, &val); err != nil {
					nfe.AddError(fmt.Errorf("failed to asn1.Unmarshal ipAddrBlocks[%d].ipAddressChoice.addressesOrRanges[%d].addressRange: %v", i, j, err))
					continue
				}
				fam.AddressRanges = append(fam.AddressRanges, IPAddressRange{Min: IPAddressPrefix(val.Min), Max: IPAddressPrefix(val.Max)})

			default:
				nfe.AddError(fmt.Errorf("unexpected ASN.1 type in ipAddrBlocks[%d].ipAddressChoice.addressesOrRanges[%d]: %+v", i, j, ar))
			}
		}
		results = append(results, &fam)
	}
	return results
}

// ASIDRange describes an inclusive range of AS Identifiers (AS numbers or routing
// domain identifiers).
type ASIDRange struct {
	Min int
	Max int
}

// ASIdentifiers describes a collection of AS Identifiers (AS numbers or routing
// domain identifiers).
type ASIdentifiers struct {
	// InheritFromIssuer indicates that the set of AS identifiers should
	// be taken from the issuer's certificate.
	InheritFromIssuer bool
	// ASIDs holds AS identifiers if InheritFromIssuer is false.
	ASIDs []int
	// ASIDs holds AS identifier ranges (inclusive) if InheritFromIssuer is false.
	ASIDRanges []ASIDRange
}

type asIdentifiers struct {
	ASNum asn1.RawValue `asn1:"optional,tag:0"`
	RDI   asn1.RawValue `asn1:"optional,tag:1"`
}

func parseASIDChoice(val asn1.RawValue, nfe *NonFatalErrors) *ASIdentifiers {
	// RFC 3779 2.3.2
	//   ASIdentifierChoice  ::= CHOICE {
	//      inherit              NULL, -- inherit from issuer --
	//      asIdsOrRanges        SEQUENCE OF ASIdOrRange }
	//   ASIdOrRange         ::= CHOICE {
	//       id                  ASId,
	//       range               ASRange }
	//   ASRange             ::= SEQUENCE {
	//       min                 ASId,
	//       max                 ASId }
	//   ASId                ::= INTEGER
	if len(val.FullBytes) == 0 { // OPTIONAL
		return nil
	}
	// ASIdentifierChoice is an ASN.1 CHOICE where the chosen alternative is indicated by (implicit)
	// tagging of the alternatives -- here, either NULL or SEQUENCE OF.
	if bytes.Equal(val.Bytes, asn1.NullBytes) {
		return &ASIdentifiers{InheritFromIssuer: true}
	}
	var ids []asn1.RawValue
	if rest, err := asn1.Unmarshal(val.Bytes, &ids); err != nil {
		nfe.AddError(fmt.Errorf("failed to asn1.Unmarshal ASIdentifiers.asIdsOrRanges: %v", err))
		return nil
	} else if len(rest) != 0 {
		nfe.AddError(errors.New("trailing data after ASIdentifiers.asIdsOrRanges"))
		return nil
	}
	var asID ASIdentifiers
	for i, id := range ids {
		// Each ASIdOrRange is a CHOICE where the alternatives have distinct (implicit)
		// tags -- here, either INTEGER or SEQUENCE.
		switch id.Tag {
		case asn1.TagInteger:
			var val int
			if _, err := asn1.Unmarshal(id.FullBytes, &val); err != nil {
				nfe.AddError(fmt.Errorf("failed to asn1.Unmarshal ASIdentifiers.asIdsOrRanges[%d].id: %v", i, err))
				continue
			}
			asID.ASIDs = append(asID.ASIDs, val)

		case asn1.TagSequence:
			var val ASIDRange
			if _, err := asn1.Unmarshal(id.FullBytes, &val); err != nil {
				nfe.AddError(fmt.Errorf("failed to asn1.Unmarshal ASIdentifiers.asIdsOrRanges[%d].range: %v", i, err))
				continue
			}
			asID.ASIDRanges = append(asID.ASIDRanges, val)

		default:
			nfe.AddError(fmt.Errorf("unexpected value in ASIdentifiers.asIdsOrRanges[%d]: %+v", i, id))
		}
	}
	return &asID
}

func parseRPKIASIdentifiers(data []byte, nfe *NonFatalErrors) (*ASIdentifiers, *ASIdentifiers) {
	// RFC 3779 2.3.2
	//   ASIdentifiers       ::= SEQUENCE {
	//       asnum               [0] EXPLICIT ASIdentifierChoice OPTIONAL,
	//       rdi                 [1] EXPLICIT ASIdentifierChoice OPTIONAL}
	var asIDs asIdentifiers
	if rest, err := asn1.Unmarshal(data, &asIDs); err != nil {
		nfe.AddError(fmt.Errorf("failed to asn1.Unmarshal ASIdentifiers extension: %v", err))
		return nil, nil
	} else if len(rest) != 0 {
		nfe.AddError(errors.New("trailing data after ASIdentifiers extension"))
		return nil, nil
	}
	return parseASIDChoice(asIDs.ASNum, nfe), parseASIDChoice(asIDs.RDI, nfe)
}
