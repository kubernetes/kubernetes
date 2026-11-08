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
	"time"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/value"
)

type numericObservation struct {
	RoundedOriginal           string `json:"roundedOriginal"`
	RoundedControl            string `json:"roundedControl"`
	OriginalEqualsDesired     bool   `json:"originalEqualsDesired"`
	DesiredEqualsControl      bool   `json:"desiredEqualsControl"`
	OriginalEqualsControl     bool   `json:"originalEqualsControl"`
	DifferenceRemainder       string `json:"differenceRemainder"`
	ExcludedContainsRemainder bool   `json:"excludedContainsRemainder"`
	ScalarOnlyAmount          string `json:"scalarOnlyAmount"`
	WithNoteAmount            string `json:"withNoteAmount"`
	IntegerConflict           bool   `json:"integerConflict"`
	BothCoOwn                 bool   `json:"bothCoOwn"`
}

func nativeNumericObservation() (numericObservation, error) {
	const original = int64(9223372036854775807)
	const control = original - 1
	const desired = float64(9223372036854775808)
	result := numericObservation{
		OriginalEqualsDesired: value.Equals(value.NewValueInterface(original), value.NewValueInterface(desired)),
		DesiredEqualsControl:  value.Equals(value.NewValueInterface(desired), value.NewValueInterface(control)),
		OriginalEqualsControl: value.Equals(value.NewValueInterface(original), value.NewValueInterface(control)),
	}
	result.RoundedOriginal, _ = numericMagnitude(float64(original))
	result.RoundedControl, _ = numericMagnitude(float64(control))

	// These individually sorted numeric-key sets expose the nontransitive comparison boundary.
	oldKey := fieldpath.KeyElementByFields("id", original)
	excluded := &fieldpath.PathElementSet{}
	excluded.Insert(fieldpath.KeyElementByFields("id", desired))
	left := &fieldpath.PathElementSet{}
	left.Insert(fieldpath.KeyElementByFields("id", control))
	left.Insert(oldKey)
	difference := left.Difference(excluded)
	if difference.Size() != 1 {
		return result, fmt.Errorf("numeric key difference changed its expected witness size: %d", difference.Size())
	}
	var remaining fieldpath.PathElement
	difference.Iterate(func(element fieldpath.PathElement) { remaining = element })
	if remaining.Key == nil || len(*remaining.Key) != 1 {
		return result, fmt.Errorf("numeric key difference returned an unexpected key")
	}
	var err error
	result.DifferenceRemainder, err = numericMagnitude((*remaining.Key)[0].Value.Unstructured())
	if err != nil {
		return result, err
	}
	result.ExcludedContainsRemainder = excluded.Has(remaining)

	c := Case{Schema: `
types:
- name: Document
  map:
    fields:
    - name: apiVersion
      type: {scalar: string}
    - name: kind
      type: {scalar: string}
    - name: amount
      type: {scalar: numeric}
    - name: note
      type: {scalar: string}
`}
	result.BothCoOwn = true
	for _, variant := range []string{"scalar-only", "with-note", "integer-control"} {
		engine, err := newGoEngine(c)
		if err != nil {
			return result, err
		}
		before, err := engine.step(Step{Manager: "A", Apply: map[string]any{"amount": original}})
		if err != nil || before.Error != "" {
			return result, fmt.Errorf("numeric A initialization failed: snapshot=%+v error=%v", before, err)
		}
		if amount, ok := engine.live.Object["amount"].(int64); !ok || amount != original {
			return result, fmt.Errorf("numeric initialization lost the exact int64")
		}
		intent := map[string]any{"amount": desired}
		if variant != "scalar-only" {
			intent["note"] = "changed"
		}
		if variant == "integer-control" {
			intent["amount"] = control
		}
		after, err := engine.step(Step{Manager: "B", Apply: intent})
		if err != nil {
			return result, err
		}
		if variant == "integer-control" {
			result.IntegerConflict = after.Error == "conflict"
			if !reflect.DeepEqual(before.Cells, after.Cells) || !reflect.DeepEqual(before.Owners, after.Owners) {
				return result, fmt.Errorf("numeric integer conflict changed application data or ownership")
			}
			continue
		}
		if after.Error != "" {
			return result, fmt.Errorf("numeric %s was unexpectedly rejected: %s", variant, after.Error)
		}
		amount, err := numericMagnitude(engine.live.Object["amount"])
		if err != nil {
			return result, err
		}
		if variant == "scalar-only" {
			result.ScalarOnlyAmount = amount
		} else {
			result.WithNoteAmount = amount
		}
		result.BothCoOwn = result.BothCoOwn &&
			slices.Contains(after.Owners["A"], ".amount") && slices.Contains(after.Owners["B"], ".amount")
	}
	return result, nil
}

// CompareNumericCounterexamples checks source-model witnesses against native primitives and managedfields.
func CompareNumericCounterexamples(ctx context.Context, model string) error {
	actual, err := nativeNumericObservation()
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	output, err := runModel(ctx, model, "--numeric-counterexamples", nil)
	if err != nil {
		return err
	}
	var lean numericObservation
	if err := json.Unmarshal(output, &lean); err != nil {
		return fmt.Errorf("invalid numeric source report: %w", err)
	}
	if actual != lean {
		return fmt.Errorf("numeric source/Go observations differ: Go=%+v Lean=%+v", actual, lean)
	}
	return nil
}
