// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
