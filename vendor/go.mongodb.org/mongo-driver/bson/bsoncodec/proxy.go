// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsoncodec

// Proxy is an interface implemented by types that cannot themselves be directly encoded. Types
// that implement this interface with have ProxyBSON called during the encoding process and that
// value will be encoded in place for the implementer.
type Proxy interface {
	ProxyBSON() (interface{}, error)
}
