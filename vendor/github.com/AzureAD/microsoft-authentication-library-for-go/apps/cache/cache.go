// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
Package cache allows third parties to implement external storage for caching token data
for distributed systems or multiple local applications access.

The data stored and extracted will represent the entire cache. Therefore it is recommended
one msal instance per user. This data is considered opaque and there are no guarantees to
implementers on the format being passed.
*/
package cache

// Marshaler marshals data from an internal cache to bytes that can be stored.
type Marshaler interface {
	Marshal() ([]byte, error)
}

// Unmarshaler unmarshals data from a storage medium into the internal cache, overwriting it.
type Unmarshaler interface {
	Unmarshal([]byte) error
}

// Serializer can serialize the cache to binary or from binary into the cache.
type Serializer interface {
	Marshaler
	Unmarshaler
}

// ExportReplace is used export or replace what is in the cache.
type ExportReplace interface {
	// Replace replaces the cache with what is in external storage.
	// key is the suggested key which can be used for partioning the cache
	Replace(cache Unmarshaler, key string)
	// Export writes the binary representation of the cache (cache.Marshal()) to
	// external storage. This is considered opaque.
	// key is the suggested key which can be used for partioning the cache
	Export(cache Marshaler, key string)
}
