// Copyright The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build go1.27

package model

import (
	"bytes"
	"encoding/json/jsontext"
	"errors"
)

var nullBytes = []byte("null")

func (s *SamplePair) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if t, err := dec.ReadToken(); err != nil {
		return err
	} else if t.Kind() == jsontext.KindNull {
		return nil
	} else if t.Kind() != jsontext.KindBeginArray {
		return errors.New("expected [")
	}

	// Loop until we see the end of the array (tolerate arrays of any length).
	for i := 0; dec.PeekKind() != jsontext.KindEndArray; i++ {
		switch i {
		case 0:
			if v, err := dec.ReadValue(); err != nil {
				return err
			} else if bytes.Equal(v, nullBytes) {
				// Skip null values to match SamplePair#UnmarshalJSON behavior.
			} else if err := s.Timestamp.UnmarshalJSON(v); err != nil {
				return err
			}
		case 1:
			if v, err := dec.ReadValue(); err != nil {
				return err
			} else if bytes.Equal(v, nullBytes) {
				// Skip null values to match SamplePair#UnmarshalJSON behavior.
			} else if err := s.Value.UnmarshalJSON(v); err != nil {
				return err
			}
		default:
			// Skip any remaining array items to match UnmarshalJSON behavior.
			if err := dec.SkipValue(); err != nil {
				return err
			}
		}
	}

	// Read the final end array token.
	if t, err := dec.ReadToken(); err != nil {
		return err
	} else if t.Kind() != jsontext.KindEndArray {
		return errors.New("expected ]")
	}

	return nil
}
