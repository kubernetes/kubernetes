// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsoncodec

import (
	"errors"
	"reflect"
	"sync"

	"go.mongodb.org/mongo-driver/bson/bsontype"
)

// ErrNilType is returned when nil is passed to either LookupEncoder or LookupDecoder.
var ErrNilType = errors.New("cannot perform a decoder lookup on <nil>")

// ErrNotPointer is returned when a non-pointer type is provided to LookupDecoder.
var ErrNotPointer = errors.New("non-pointer provided to LookupDecoder")

// ErrNoEncoder is returned when there wasn't an encoder available for a type.
type ErrNoEncoder struct {
	Type reflect.Type
}

func (ene ErrNoEncoder) Error() string {
	if ene.Type == nil {
		return "no encoder found for <nil>"
	}
	return "no encoder found for " + ene.Type.String()
}

// ErrNoDecoder is returned when there wasn't a decoder available for a type.
type ErrNoDecoder struct {
	Type reflect.Type
}

func (end ErrNoDecoder) Error() string {
	return "no decoder found for " + end.Type.String()
}

// ErrNoTypeMapEntry is returned when there wasn't a type available for the provided BSON type.
type ErrNoTypeMapEntry struct {
	Type bsontype.Type
}

func (entme ErrNoTypeMapEntry) Error() string {
	return "no type map entry found for " + entme.Type.String()
}

// ErrNotInterface is returned when the provided type is not an interface.
var ErrNotInterface = errors.New("The provided type is not an interface")

var defaultRegistry *Registry

func init() {
	defaultRegistry = buildDefaultRegistry()
}

// A RegistryBuilder is used to build a Registry. This type is not goroutine
// safe.
type RegistryBuilder struct {
	typeEncoders      map[reflect.Type]ValueEncoder
	interfaceEncoders []interfaceValueEncoder
	kindEncoders      map[reflect.Kind]ValueEncoder

	typeDecoders      map[reflect.Type]ValueDecoder
	interfaceDecoders []interfaceValueDecoder
	kindDecoders      map[reflect.Kind]ValueDecoder

	typeMap map[bsontype.Type]reflect.Type
}

// A Registry is used to store and retrieve codecs for types and interfaces. This type is the main
// typed passed around and Encoders and Decoders are constructed from it.
type Registry struct {
	typeEncoders map[reflect.Type]ValueEncoder
	typeDecoders map[reflect.Type]ValueDecoder

	interfaceEncoders []interfaceValueEncoder
	interfaceDecoders []interfaceValueDecoder

	kindEncoders map[reflect.Kind]ValueEncoder
	kindDecoders map[reflect.Kind]ValueDecoder

	typeMap map[bsontype.Type]reflect.Type

	mu sync.RWMutex
}

// NewRegistryBuilder creates a new empty RegistryBuilder.
func NewRegistryBuilder() *RegistryBuilder {
	return &RegistryBuilder{
		typeEncoders: make(map[reflect.Type]ValueEncoder),
		typeDecoders: make(map[reflect.Type]ValueDecoder),

		interfaceEncoders: make([]interfaceValueEncoder, 0),
		interfaceDecoders: make([]interfaceValueDecoder, 0),

		kindEncoders: make(map[reflect.Kind]ValueEncoder),
		kindDecoders: make(map[reflect.Kind]ValueDecoder),

		typeMap: make(map[bsontype.Type]reflect.Type),
	}
}

func buildDefaultRegistry() *Registry {
	rb := NewRegistryBuilder()
	defaultValueEncoders.RegisterDefaultEncoders(rb)
	defaultValueDecoders.RegisterDefaultDecoders(rb)
	return rb.Build()
}

// RegisterCodec will register the provided ValueCodec for the provided type.
func (rb *RegistryBuilder) RegisterCodec(t reflect.Type, codec ValueCodec) *RegistryBuilder {
	rb.RegisterEncoder(t, codec)
	rb.RegisterDecoder(t, codec)
	return rb
}

// RegisterEncoder will register the provided ValueEncoder to the provided type.
//
// The type registered will be used directly, so an encoder can be registered for a type and a
// different encoder can be registered for a pointer to that type.
func (rb *RegistryBuilder) RegisterEncoder(t reflect.Type, enc ValueEncoder) *RegistryBuilder {
	if t == tEmpty {
		rb.typeEncoders[t] = enc
		return rb
	}
	switch t.Kind() {
	case reflect.Interface:
		for idx, ir := range rb.interfaceEncoders {
			if ir.i == t {
				rb.interfaceEncoders[idx].ve = enc
				return rb
			}
		}

		rb.interfaceEncoders = append(rb.interfaceEncoders, interfaceValueEncoder{i: t, ve: enc})
	default:
		rb.typeEncoders[t] = enc
	}
	return rb
}

// RegisterDecoder will register the provided ValueDecoder to the provided type.
//
// The type registered will be used directly, so a decoder can be registered for a type and a
// different decoder can be registered for a pointer to that type.
func (rb *RegistryBuilder) RegisterDecoder(t reflect.Type, dec ValueDecoder) *RegistryBuilder {
	if t == nil {
		rb.typeDecoders[nil] = dec
		return rb
	}
	if t == tEmpty {
		rb.typeDecoders[t] = dec
		return rb
	}
	switch t.Kind() {
	case reflect.Interface:
		for idx, ir := range rb.interfaceDecoders {
			if ir.i == t {
				rb.interfaceDecoders[idx].vd = dec
				return rb
			}
		}

		rb.interfaceDecoders = append(rb.interfaceDecoders, interfaceValueDecoder{i: t, vd: dec})
	default:
		rb.typeDecoders[t] = dec
	}
	return rb
}

// RegisterDefaultEncoder will registr the provided ValueEncoder to the provided
// kind.
func (rb *RegistryBuilder) RegisterDefaultEncoder(kind reflect.Kind, enc ValueEncoder) *RegistryBuilder {
	rb.kindEncoders[kind] = enc
	return rb
}

// RegisterDefaultDecoder will register the provided ValueDecoder to the
// provided kind.
func (rb *RegistryBuilder) RegisterDefaultDecoder(kind reflect.Kind, dec ValueDecoder) *RegistryBuilder {
	rb.kindDecoders[kind] = dec
	return rb
}

// RegisterTypeMapEntry will register the provided type to the BSON type. The primary usage for this
// mapping is decoding situations where an empty interface is used and a default type needs to be
// created and decoded into.
//
// NOTE: It is unlikely that registering a type for BSON Embedded Document is actually desired. By
// registering a type map entry for BSON Embedded Document the type registered will be used in any
// case where a BSON Embedded Document will be decoded into an empty interface. For example, if you
// register primitive.M, the EmptyInterface decoder will always use primitive.M, even if an ancestor
// was a primitive.D.
func (rb *RegistryBuilder) RegisterTypeMapEntry(bt bsontype.Type, rt reflect.Type) *RegistryBuilder {
	rb.typeMap[bt] = rt
	return rb
}

// Build creates a Registry from the current state of this RegistryBuilder.
func (rb *RegistryBuilder) Build() *Registry {
	registry := new(Registry)

	registry.typeEncoders = make(map[reflect.Type]ValueEncoder)
	for t, enc := range rb.typeEncoders {
		registry.typeEncoders[t] = enc
	}

	registry.typeDecoders = make(map[reflect.Type]ValueDecoder)
	for t, dec := range rb.typeDecoders {
		registry.typeDecoders[t] = dec
	}

	registry.interfaceEncoders = make([]interfaceValueEncoder, len(rb.interfaceEncoders))
	copy(registry.interfaceEncoders, rb.interfaceEncoders)

	registry.interfaceDecoders = make([]interfaceValueDecoder, len(rb.interfaceDecoders))
	copy(registry.interfaceDecoders, rb.interfaceDecoders)

	registry.kindEncoders = make(map[reflect.Kind]ValueEncoder)
	for kind, enc := range rb.kindEncoders {
		registry.kindEncoders[kind] = enc
	}

	registry.kindDecoders = make(map[reflect.Kind]ValueDecoder)
	for kind, dec := range rb.kindDecoders {
		registry.kindDecoders[kind] = dec
	}

	registry.typeMap = make(map[bsontype.Type]reflect.Type)
	for bt, rt := range rb.typeMap {
		registry.typeMap[bt] = rt
	}

	return registry
}

// LookupEncoder will inspect the registry for an encoder that satisfies the
// type provided. An encoder registered for a specific type will take
// precedence over an encoder registered for an interface the type satisfies,
// which takes precedence over an encoder for the reflect.Kind of the value. If
// no encoder can be found, an error is returned.
func (r *Registry) LookupEncoder(t reflect.Type) (ValueEncoder, error) {
	encodererr := ErrNoEncoder{Type: t}
	r.mu.RLock()
	enc, found := r.lookupTypeEncoder(t)
	r.mu.RUnlock()
	if found {
		if enc == nil {
			return nil, ErrNoEncoder{Type: t}
		}
		return enc, nil
	}

	enc, found = r.lookupInterfaceEncoder(t)
	if found {
		r.mu.Lock()
		r.typeEncoders[t] = enc
		r.mu.Unlock()
		return enc, nil
	}

	if t == nil {
		r.mu.Lock()
		r.typeEncoders[t] = nil
		r.mu.Unlock()
		return nil, encodererr
	}

	enc, found = r.kindEncoders[t.Kind()]
	if !found {
		r.mu.Lock()
		r.typeEncoders[t] = nil
		r.mu.Unlock()
		return nil, encodererr
	}

	r.mu.Lock()
	r.typeEncoders[t] = enc
	r.mu.Unlock()
	return enc, nil
}

func (r *Registry) lookupTypeEncoder(t reflect.Type) (ValueEncoder, bool) {
	enc, found := r.typeEncoders[t]
	return enc, found
}

func (r *Registry) lookupInterfaceEncoder(t reflect.Type) (ValueEncoder, bool) {
	if t == nil {
		return nil, false
	}
	for _, ienc := range r.interfaceEncoders {
		if !t.Implements(ienc.i) {
			continue
		}

		return ienc.ve, true
	}
	return nil, false
}

// LookupDecoder will inspect the registry for a decoder that satisfies the
// type provided. A decoder registered for a specific type will take
// precedence over a decoder registered for an interface the type satisfies,
// which takes precedence over a decoder for the reflect.Kind of the value. If
// no decoder can be found, an error is returned.
func (r *Registry) LookupDecoder(t reflect.Type) (ValueDecoder, error) {
	if t == nil {
		return nil, ErrNilType
	}
	decodererr := ErrNoDecoder{Type: t}
	r.mu.RLock()
	dec, found := r.lookupTypeDecoder(t)
	r.mu.RUnlock()
	if found {
		if dec == nil {
			return nil, ErrNoDecoder{Type: t}
		}
		return dec, nil
	}

	dec, found = r.lookupInterfaceDecoder(t)
	if found {
		r.mu.Lock()
		r.typeDecoders[t] = dec
		r.mu.Unlock()
		return dec, nil
	}

	dec, found = r.kindDecoders[t.Kind()]
	if !found {
		r.mu.Lock()
		r.typeDecoders[t] = nil
		r.mu.Unlock()
		return nil, decodererr
	}

	r.mu.Lock()
	r.typeDecoders[t] = dec
	r.mu.Unlock()
	return dec, nil
}

func (r *Registry) lookupTypeDecoder(t reflect.Type) (ValueDecoder, bool) {
	dec, found := r.typeDecoders[t]
	return dec, found
}

func (r *Registry) lookupInterfaceDecoder(t reflect.Type) (ValueDecoder, bool) {
	for _, idec := range r.interfaceDecoders {
		if !t.Implements(idec.i) && !reflect.PtrTo(t).Implements(idec.i) {
			continue
		}

		return idec.vd, true
	}
	return nil, false
}

// LookupTypeMapEntry inspects the registry's type map for a Go type for the corresponding BSON
// type. If no type is found, ErrNoTypeMapEntry is returned.
func (r *Registry) LookupTypeMapEntry(bt bsontype.Type) (reflect.Type, error) {
	t, ok := r.typeMap[bt]
	if !ok || t == nil {
		return nil, ErrNoTypeMapEntry{Type: bt}
	}
	return t, nil
}

type interfaceValueEncoder struct {
	i  reflect.Type
	ve ValueEncoder
}

type interfaceValueDecoder struct {
	i  reflect.Type
	vd ValueDecoder
}
