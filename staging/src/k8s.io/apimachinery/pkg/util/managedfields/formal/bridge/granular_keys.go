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
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"strconv"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

type granularKeyObservation struct {
	InitialDataAccepted            bool `json:"initialDataAccepted"`
	RawOwnerDoesNotClaimNeighbor   bool `json:"rawOwnerDoesNotClaimNeighbor"`
	FalseConflict                  bool `json:"falseConflict"`
	SmallIndependentAccepted       bool `json:"smallIndependentAccepted"`
	ForceKeepsPayloadButDropsOwner bool `json:"forceKeepsPayloadButDropsOwner"`
}

func compareGranularKeyCounterexamples(ctx context.Context, model string) error {
	const original = int64(9007199254740993)
	const numericKeySchema = `types:
- name: Document
  map:
    fields:
    - name: endpoints
      type:
        list:
          elementType: {namedType: Endpoint}
          elementRelationship: associative
          keys: [id]
- name: Endpoint
  map:
    fields:
    - name: id
      type: {scalar: numeric}
    - name: payload
      type: {scalar: string}
`
	c := Case{Schema: numericKeySchema}
	rawPath := fieldpath.MakePathOrDie("endpoints", fieldpath.KeyByFields("id", original))
	neighborPath := fieldpath.MakePathOrDie("endpoints", fieldpath.KeyByFields("id", original-1))
	// Both operands are exact integers before decoding; the mixed comparator is not used here.
	actual := granularKeyObservation{RawOwnerDoesNotClaimNeighbor: !fieldpath.NewSet(rawPath).Has(neighborPath)}
	for _, key := range []int64{original, 42} {
		engine, err := newGoEngine(c)
		if err != nil {
			return err
		}
		initial := map[string]any{"endpoints": []any{map[string]any{"id": key, "payload": "old"}}}
		intent := map[string]any{"endpoints": []any{map[string]any{"id": key - 1, "payload": "independent"}}}
		initialError, err := engine.advance(Step{Manager: "A", Apply: initial})
		if err != nil {
			return err
		}
		before, err := observeNativeKeyState(engine)
		if err != nil {
			return err
		}
		expectedClaimKey := key
		if key == original {
			expectedClaimKey--
		}
		if initialError != "" || !reflect.DeepEqual(before.Data, map[int64]string{key: "old"}) ||
			!slices.Contains(before.PayloadClaims["A"], strconv.FormatInt(expectedClaimKey, 10)) {
			return fmt.Errorf("integer-key initialization changed exact application data or ownership: %+v", before)
		}
		requestError, err := engine.advance(Step{Manager: "B", Apply: intent})
		if err != nil {
			return err
		}
		after, err := observeNativeKeyState(engine)
		if err != nil {
			return err
		}
		if key == original {
			actual.InitialDataAccepted = true
			actual.FalseConflict = requestError == "conflict"
			if actual.FalseConflict && !reflect.DeepEqual(before, after) {
				return fmt.Errorf("false integer-key conflict changed native application data or fieldsets")
			}
			requestError, err = engine.advance(Step{Manager: "B", Apply: intent, Force: true})
			if err != nil {
				return err
			}
			after, err = observeNativeKeyState(engine)
			if err != nil {
				return err
			}
		}
		correctData := requestError == "" && reflect.DeepEqual(after.Data, map[int64]string{key: "old", key - 1: "independent"})
		if key == original {
			actual.ForceKeepsPayloadButDropsOwner = correctData && !after.EndpointClaims["A"]
		} else {
			actual.SmallIndependentAccepted = correctData &&
				slices.Contains(after.Memberships["A"], "42") && slices.Contains(after.Memberships["B"], "41")
		}
	}
	output, err := runModel(ctx, model, "--granular-key-counterexamples", nil)
	if err != nil {
		return err
	}
	var lean granularKeyObservation
	if err := json.Unmarshal(output, &lean); err != nil {
		return fmt.Errorf("invalid granular key source report: %w", err)
	}
	if actual != lean {
		return fmt.Errorf("granular key source/Go observations differ: Go=%+v Lean=%+v", actual, lean)
	}
	return nil
}

// Native observations remain available when broken ownership cannot inhabit the normalized model.
type nativeKeyState struct {
	Data           map[int64]string
	Fields         map[string]nativeFieldset
	EndpointClaims map[string]bool
	Memberships    map[string][]string
	PayloadClaims  map[string][]string
}

type nativeFieldset struct {
	Version string
	Applied bool
	JSON    string
}

func observeNativeKeyState(engine *goEngine) (nativeKeyState, error) {
	state := nativeKeyState{Data: map[int64]string{}, Fields: map[string]nativeFieldset{}, EndpointClaims: map[string]bool{}, Memberships: map[string][]string{}, PayloadClaims: map[string][]string{}}
	object, ok := engine.rawObject().(map[string]any)
	if !ok {
		return state, fmt.Errorf("native key witness lost the object envelope")
	}
	items, ok := object["endpoints"].([]any)
	if !ok {
		return state, fmt.Errorf("native key witness lost its endpoints list")
	}
	for _, value := range items {
		item, ok := value.(map[string]any)
		if !ok {
			return state, fmt.Errorf("native key witness contains non-object item %T", value)
		}
		key, ok := item["id"].(int64)
		if !ok {
			return state, fmt.Errorf("native data key lost its exact int64 type: %T", item["id"])
		}
		payload, ok := item["payload"].(string)
		if !ok {
			return state, fmt.Errorf("native key payload has type %T", item["payload"])
		}
		if _, duplicate := state.Data[key]; duplicate {
			return state, fmt.Errorf("native key witness contains duplicate key %d", key)
		}
		state.Data[key] = payload
	}
	fields, err := engine.observedFields()
	if err != nil {
		return state, err
	}
	for actor, entry := range fields {
		wire, err := entry.Set().ToJSON()
		if err != nil {
			return state, err
		}
		state.Fields[actor] = nativeFieldset{Version: string(entry.APIVersion()), Applied: entry.Applied(), JSON: string(wire)}
		var invalid error
		entry.Set().Iterate(func(path fieldpath.Path) {
			if len(path) == 0 || path[0].FieldName == nil || *path[0].FieldName != "endpoints" {
				return
			}
			state.EndpointClaims[actor] = true
			payload := len(path) == 3 && path[2].FieldName != nil && *path[2].FieldName == "payload"
			if len(path) != 2 && !payload {
				return
			}
			if path[1].Key == nil || len(*path[1].Key) != 1 || (*path[1].Key)[0].Name != "id" {
				invalid = fmt.Errorf("native key ownership has unexpected selector %v", path)
				return
			}
			magnitude, err := numericMagnitude((*path[1].Key)[0].Value.Unstructured())
			if err != nil {
				invalid = err
				return
			}
			if payload {
				state.PayloadClaims[actor] = append(state.PayloadClaims[actor], magnitude)
			} else {
				state.Memberships[actor] = append(state.Memberships[actor], magnitude)
			}
		})
		if invalid != nil {
			return state, invalid
		}
		sort.Strings(state.Memberships[actor])
		sort.Strings(state.PayloadClaims[actor])
	}
	return state, nil
}
