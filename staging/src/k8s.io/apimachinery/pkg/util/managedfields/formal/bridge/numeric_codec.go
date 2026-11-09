/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package bridge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

type keyCodecObservation struct {
	OriginalKey       string `json:"originalKey"`
	DecodedKey        string `json:"decodedKey"`
	RetainedOriginal  bool   `json:"retainedOriginal"`
	ClaimsNeighbor    bool   `json:"claimsNeighbor"`
	DistinctIntegers  bool   `json:"distinctIntegers"`
	SmallKeyPreserved bool   `json:"smallKeyPreserved"`
}

// The witness floats are integral, so fixed decimal formatting preserves their exact mathematical magnitude.
func numericMagnitude(number any) (string, error) {
	switch number := number.(type) {
	case int64:
		return strconv.FormatInt(number, 10), nil
	case float64:
		return strconv.FormatFloat(number, 'f', 0, 64), nil
	default:
		return "", fmt.Errorf("numeric witness has unsupported type %T", number)
	}
}

// Inspect exact native key magnitudes: Has/Equals use the broken mixed comparison and cannot be an oracle here.
func roundtripNumericKey(number int64) (string, error) {
	set := fieldpath.NewSet(fieldpath.MakePathOrDie("endpoints", fieldpath.KeyByFields("id", number)))
	wire, err := set.ToJSON()
	if err != nil {
		return "", err
	}
	var restored fieldpath.Set
	if err := restored.FromJSON(bytes.NewReader(wire)); err != nil {
		return "", err
	}
	var decoded string
	count := 0
	restored.Iterate(func(path fieldpath.Path) {
		count++
		if len(path) != 2 || path[0].FieldName == nil || *path[0].FieldName != "endpoints" ||
			path[1].Key == nil || len(*path[1].Key) != 1 || (*path[1].Key)[0].Name != "id" {
			err = fmt.Errorf("numeric key round trip returned an unexpected path: %v", path)
			return
		}
		decoded, err = numericMagnitude((*path[1].Key)[0].Value.Unstructured())
	})
	if err != nil {
		return "", err
	}
	if count != 1 {
		return "", fmt.Errorf("numeric key round trip returned %d claims; expected one", count)
	}
	reserialized, err := restored.ToJSON()
	if err != nil {
		return "", err
	}
	if bytes.Equal(wire, reserialized) != (decoded == strconv.FormatInt(number, 10)) {
		return "", fmt.Errorf("numeric key wire round trip disagrees with exact decoded magnitude")
	}
	return decoded, nil
}

// CompareNumericCodecCounterexamples runs on the vendored dependency and both pinned upstream revisions.
func CompareNumericCodecCounterexamples(ctx context.Context, model string) error {
	const original = int64(9007199254740993)
	decoded, err := roundtripNumericKey(original)
	if err != nil {
		return err
	}
	small, err := roundtripNumericKey(42)
	if err != nil {
		return err
	}
	actual := keyCodecObservation{
		OriginalKey: strconv.FormatInt(original, 10), DecodedKey: decoded,
		RetainedOriginal: decoded == strconv.FormatInt(original, 10),
		ClaimsNeighbor:   decoded == strconv.FormatInt(original-1, 10),
		DistinctIntegers: original != original-1, SmallKeyPreserved: small == "42",
	}
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	output, err := runModel(ctx, model, "--numeric-codec-counterexamples", nil)
	if err != nil {
		return err
	}
	var lean keyCodecObservation
	if err := json.Unmarshal(output, &lean); err != nil {
		return fmt.Errorf("invalid numeric codec report: %w", err)
	}
	if actual != lean {
		return fmt.Errorf("numeric codec/Go observations differ: Go=%+v Lean=%+v", actual, lean)
	}
	return compareGranularKeyCounterexamples(ctx, model)
}
