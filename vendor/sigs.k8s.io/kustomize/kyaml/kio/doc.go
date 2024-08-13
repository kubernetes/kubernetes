// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package kio contains libraries for reading and writing collections of Resources.
//
// Reading Resources
//
// Resources are Read using a kio.Reader function.  Examples:
//  [kio.LocalPackageReader{}, kio.ByteReader{}]
//
// Resources read using a LocalPackageReader will have annotations applied so they can be
// written back to the files they were read from.
//
// Modifying Resources
//
// Resources are modified using a kio.Filter.  The kio.Filter accepts a collection of
// Resources as input, and returns a new collection as output.
// It is recommended to use the yaml package for manipulating individual Resources in
// the collection.
//
// Writing Resources
//
// Resources are Read using a kio.Reader function.  Examples:
//  [kio.LocalPackageWriter{}, kio.ByteWriter{}]
//
// ReadWriters
//
// It is preferred to use a ReadWriter when reading and writing from / to the same source.
//
// Building Pipelines
//
// The preferred way to transforms a collection of Resources is to use kio.Pipeline to Read,
// Modify and Write the collection of Resources.  Pipeline will automatically sequentially
// invoke the Read, Modify, Write steps, returning and error immediately on any failure.
package kio
