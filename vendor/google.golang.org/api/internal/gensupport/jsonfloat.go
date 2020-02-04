// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

// JSONFloat64 is a float64 that supports proper unmarshaling of special float
// values in JSON, according to
// https://developers.google.com/protocol-buffers/docs/proto3#json. Although
// that is a proto-to-JSON spec, it applies to all Google APIs.
//
// The jsonpb package
// (https://github.com/golang/protobuf/blob/master/jsonpb/jsonpb.go) has
// similar functionality, but only for direct translation from proto messages
// to JSON.
type JSONFloat64 float64

func (f *JSONFloat64) UnmarshalJSON(data []byte) error {
	var ff float64
	if err := json.Unmarshal(data, &ff); err == nil {
		*f = JSONFloat64(ff)
		return nil
	}
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		switch s {
		case "NaN":
			ff = math.NaN()
		case "Infinity":
			ff = math.Inf(1)
		case "-Infinity":
			ff = math.Inf(-1)
		default:
			return fmt.Errorf("google.golang.org/api/internal: bad float string %q", s)
		}
		*f = JSONFloat64(ff)
		return nil
	}
	return errors.New("google.golang.org/api/internal: data not float or string")
}
