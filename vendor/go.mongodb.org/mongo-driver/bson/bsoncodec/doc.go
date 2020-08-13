// Package bsoncodec provides a system for encoding values to BSON representations and decoding
// values from BSON representations. This package considers both binary BSON and ExtendedJSON as
// BSON representations. The types in this package enable a flexible system for handling this
// encoding and decoding.
//
// The codec system is composed of two parts:
//
// 1) ValueEncoders and ValueDecoders that handle encoding and decoding Go values to and from BSON
// representations.
//
// 2) A Registry that holds these ValueEncoders and ValueDecoders and provides methods for
// retrieving them.
//
// ValueEncoders and ValueDecoders
//
// The ValueEncoder interface is implemented by types that can encode a provided Go type to BSON.
// The value to encode is provided as a reflect.Value and a bsonrw.ValueWriter is used within the
// EncodeValue method to actually create the BSON representation. For convenience, ValueEncoderFunc
// is provided to allow use of a function with the correct signature as a ValueEncoder. An
// EncodeContext instance is provided to allow implementations to lookup further ValueEncoders and
// to provide configuration information.
//
// The ValueDecoder interface is the inverse of the ValueEncoder. Implementations should ensure that
// the value they receive is settable. Similar to ValueEncoderFunc, ValueDecoderFunc is provided to
// allow the use of a function with the correct signature as a ValueDecoder. A DecodeContext
// instance is provided and serves similar functionality to the EncodeContext.
//
// Registry and RegistryBuilder
//
// A Registry is an immutable store for ValueEncoders, ValueDecoders, and a type map. For looking up
// ValueEncoders and Decoders the Registry first attempts to find a ValueEncoder or ValueDecoder for
// the type provided; if one cannot be found it then checks to see if a registered ValueEncoder or
// ValueDecoder exists for an interface the type implements. Finally, the reflect.Kind of the type
// is used to lookup a default ValueEncoder or ValueDecoder for that kind. If no ValueEncoder or
// ValueDecoder can be found, an error is returned.
//
// The Registry also holds a type map. This allows users to retrieve the Go type that should be used
// when decoding a BSON value into an empty interface. This is primarily only used for the empty
// interface ValueDecoder.
//
// A RegistryBuilder is used to construct a Registry. The Register methods are used to associate
// either a reflect.Type or a reflect.Kind with a ValueEncoder or ValueDecoder. A RegistryBuilder
// returned from NewRegistryBuilder contains no registered ValueEncoders nor ValueDecoders and
// contains an empty type map.
//
// The RegisterTypeMapEntry method handles associating a BSON type with a Go type. For example, if
// you want to decode BSON int64 and int32 values into Go int instances, you would do the following:
//
//  var regbuilder *RegistryBuilder = ... intType := reflect.TypeOf(int(0))
//  regbuilder.RegisterTypeMapEntry(bsontype.Int64, intType).RegisterTypeMapEntry(bsontype.Int32,
//  intType)
//
// DefaultValueEncoders and DefaultValueDecoders
//
// The DefaultValueEncoders and DefaultValueDecoders types provide a full set of ValueEncoders and
// ValueDecoders for handling a wide range of Go types, including all of the types within the
// primitive package. To make registering these codecs easier, a helper method on each type is
// provided. For the DefaultValueEncoders type the method is called RegisterDefaultEncoders and for
// the DefaultValueDecoders type the method is called RegisterDefaultDecoders, this method also
// handles registering type map entries for each BSON type.
package bsoncodec
