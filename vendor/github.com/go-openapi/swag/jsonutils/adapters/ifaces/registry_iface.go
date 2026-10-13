// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package ifaces

import (
	"strings"
)

// Capability indicates what a JSON adapter is capable of.
type Capability uint8

const (
	CapabilityMarshalJSON Capability = 1 << iota
	CapabilityUnmarshalJSON
	CapabilityOrderedMarshalJSON
	CapabilityOrderedUnmarshalJSON
	CapabilityOrderedMap
)

func (c Capability) String() string {
	switch c {
	case CapabilityMarshalJSON:
		return "MarshalJSON"
	case CapabilityUnmarshalJSON:
		return "UnmarshalJSON"
	case CapabilityOrderedMarshalJSON:
		return "OrderedMarshalJSON"
	case CapabilityOrderedUnmarshalJSON:
		return "OrderedUnmarshalJSON"
	case CapabilityOrderedMap:
		return "OrderedMap"
	default:
		return "<unknown>"
	}
}

// Capabilities holds several unitary capability flags
type Capabilities uint8

// Has some capability flag enabled.
func (c Capabilities) Has(capability Capability) bool {
	return Capability(c)&capability > 0
}

func (c Capabilities) String() string {
	var w strings.Builder

	first := true
	for _, capability := range []Capability{
		CapabilityMarshalJSON,
		CapabilityUnmarshalJSON,
		CapabilityOrderedMarshalJSON,
		CapabilityOrderedUnmarshalJSON,
		CapabilityOrderedMap,
	} {
		if c.Has(capability) {
			if !first {
				w.WriteByte('|')
			} else {
				first = false
			}
			w.WriteString(capability.String())
		}
	}

	return w.String()
}

const (
	AllCapabilities Capabilities = Capabilities(uint8(CapabilityMarshalJSON) |
		uint8(CapabilityUnmarshalJSON) |
		uint8(CapabilityOrderedMarshalJSON) |
		uint8(CapabilityOrderedUnmarshalJSON) |
		uint8(CapabilityOrderedMap))

	AllUnorderedCapabilities Capabilities = Capabilities(uint8(CapabilityMarshalJSON) | uint8(CapabilityUnmarshalJSON))
)

// RegistryEntry describes how any given adapter registers its capabilities to the [Registrar].
type RegistryEntry struct {
	Who         string
	What        Capabilities
	Constructor func() Adapter
	Support     func(what Capability, value any) bool
}

// Registrar is a type that knows how to keep registration calls from adapters.
type Registrar interface {
	RegisterFor(RegistryEntry)
}
