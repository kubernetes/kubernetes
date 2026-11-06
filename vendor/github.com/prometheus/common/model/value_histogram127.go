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
	"encoding/json/v2"
	"errors"
)

func (s *SampleHistogramPair) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if t, err := dec.ReadToken(); err != nil {
		return err
	} else if t.Kind() != jsontext.KindBeginArray {
		return errors.New("expected [")
	}

	if v, err := dec.ReadValue(); err != nil {
		return err
	} else if bytes.Equal(v, nullBytes) {
		// Skip null values to match SamplePair#UnmarshalJSON behavior.
	} else if err := s.Timestamp.UnmarshalJSON(v); err != nil {
		return err
	}

	if err := json.UnmarshalDecode(dec, &s.Histogram); err != nil {
		return err
	} else if s.Histogram == nil {
		return errors.New("histogram is nil")
	}

	if t, err := dec.ReadToken(); err != nil {
		return err
	} else if t.Kind() != jsontext.KindEndArray {
		return errors.New("expected ]")
	}
	return nil
}

func (s *HistogramBucket) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if t, err := dec.ReadToken(); err != nil {
		return err
	} else if t.Kind() != jsontext.KindBeginArray {
		return errors.New("expected [")
	}

	if err := json.UnmarshalDecode(dec, &s.Boundaries); err != nil {
		return err
	}

	if v, err := dec.ReadValue(); err != nil {
		return err
	} else if bytes.Equal(v, nullBytes) {
		// Skip null values to match SamplePair#UnmarshalJSON behavior.
	} else if err := s.Lower.UnmarshalJSON(v); err != nil {
		return err
	}

	if v, err := dec.ReadValue(); err != nil {
		return err
	} else if bytes.Equal(v, nullBytes) {
		// Skip null values to match SamplePair#UnmarshalJSON behavior.
	} else if err := s.Upper.UnmarshalJSON(v); err != nil {
		return err
	}

	if v, err := dec.ReadValue(); err != nil {
		return err
	} else if bytes.Equal(v, nullBytes) {
		// Skip null values to match SamplePair#UnmarshalJSON behavior.
	} else if err := s.Count.UnmarshalJSON(v); err != nil {
		return err
	}

	if t, err := dec.ReadToken(); err != nil {
		return err
	} else if t.Kind() != jsontext.KindEndArray {
		return errors.New("expected ]")
	}
	return nil
}
