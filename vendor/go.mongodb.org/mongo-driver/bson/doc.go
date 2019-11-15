// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

// Package bson is a library for reading, writing, and manipulating BSON. The
// library has two families of types for representing BSON.
//
// The Raw family of types is used to validate and retrieve elements from a slice of bytes. This
// type is most useful when you want do lookups on BSON bytes without unmarshaling it into another
// type.
//
// Example:
// 		var raw bson.Raw = ... // bytes from somewhere
// 		err := raw.Validate()
// 		if err != nil { return err }
// 		val := raw.Lookup("foo")
// 		i32, ok := val.Int32OK()
// 		// do something with i32...
//
// The D family of types is used to build concise representations of BSON using native Go types.
// These types do not support automatic lookup.
//
// Example:
// 		bson.D{{"foo", "bar"}, {"hello", "world"}, {"pi", 3.14159}}
//
//
// Marshaling and Unmarshaling are handled with the Marshal and Unmarshal family of functions. If
// you need to write or read BSON from a non-slice source, an Encoder or Decoder can be used with a
// bsonrw.ValueWriter or bsonrw.ValueReader.
//
// Example:
// 		b, err := bson.Marshal(bson.D{{"foo", "bar"}})
// 		if err != nil { return err }
// 		var fooer struct {
// 			Foo string
// 		}
// 		err = bson.Unmarshal(b, &fooer)
// 		if err != nil { return err }
// 		// do something with fooer...
package bson
