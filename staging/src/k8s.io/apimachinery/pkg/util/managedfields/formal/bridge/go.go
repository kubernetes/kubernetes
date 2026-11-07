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
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields/internal"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/merge"
	"sigs.k8s.io/structured-merge-diff/v6/typed"
)

//go:embed schema.yaml
var schemaYAML string

var groupVersion = schema.GroupVersion{Group: "formal.ssa", Version: "v1"}

type typeConverter struct {
	object typed.ParseableType
}

func (c typeConverter) ObjectToTyped(obj runtime.Object, options ...typed.ValidationOptions) (*typed.TypedValue, error) {
	return c.object.FromUnstructured(obj.(*unstructured.Unstructured).Object, options...)
}

func (c typeConverter) TypedToObject(tv *typed.TypedValue) (runtime.Object, error) {
	object, ok := tv.AsValue().Unstructured().(map[string]any)
	if !ok {
		return nil, fmt.Errorf("SMD returned a non-object value")
	}
	return &unstructured.Unstructured{Object: object}, nil
}

// The bridge deliberately uses one API version and no defaulting. Conversion
// and defaulting require separate API server integration obligations.
type versionConverter struct{}

func (versionConverter) Convert(_, _ any, _ any) error {
	return fmt.Errorf("the SSA bridge does not support cross-version conversion")
}

func (versionConverter) ConvertToVersion(obj runtime.Object, _ runtime.GroupVersioner) (runtime.Object, error) {
	return obj.DeepCopyObject(), nil
}

func (versionConverter) ConvertFieldLabel(schema.GroupVersionKind, string, string) (string, string, error) {
	return "", "", fmt.Errorf("the SSA bridge does not convert field labels")
}

type noDefaults struct{}

func (noDefaults) Default(runtime.Object) {}

type objectCreator struct{}

func (objectCreator) New(gvk schema.GroupVersionKind) (runtime.Object, error) {
	object := &unstructured.Unstructured{Object: map[string]any{}}
	object.SetGroupVersionKind(gvk)
	return object, nil
}

type goEngine struct {
	manager internal.Manager
	live    *unstructured.Unstructured
	fields  internal.Managed
	schema  *structuralSchema
}

func newGoEngine(cases ...Case) (*goEngine, error) {
	c := Case{}
	if len(cases) > 0 {
		c = cases[0]
	}
	sc, err := schemaFor(c)
	if err != nil {
		return nil, err
	}
	parser, err := typed.NewParser(typed.YAMLObject(sc.legacy))
	if err != nil {
		return nil, err
	}
	manager, err := internal.NewStructuredMergeManager(
		typeConverter{parser.Type(sc.root)}, versionConverter{}, noDefaults{},
		groupVersion, groupVersion, nil)
	if err != nil {
		return nil, err
	}
	manager = internal.NewBuildManagerInfoManager(internal.NewStripMetaManager(manager), groupVersion, "")
	if c.Bootstrap {
		manager = internal.NewSkipNonAppliedManager(manager, objectCreator{})
	}
	return &goEngine{manager, envelope(map[string]any{}), internal.NewEmptyManaged(), sc}, nil
}

func envelope(object map[string]any) *unstructured.Unstructured {
	result := &unstructured.Unstructured{Object: runtime.DeepCopyJSON(object)}
	result.SetAPIVersion(groupVersion.String())
	result.SetKind("Document")
	return result
}

func (e *goEngine) step(step Step) (Snapshot, error) {
	// Updater works on a proposed managed-field map. Commit object and ownership
	// together only after success, matching the reference model's commit wrapper.
	working := internal.NewManaged(e.fields.Fields().Copy(), nil)
	var object runtime.Object
	var fields internal.Managed
	var err error
	if step.Operation == "update" {
		object, fields, err = e.manager.Update(e.live, envelope(step.Apply), working, step.Manager)
	} else {
		object, fields, err = e.manager.Apply(e.live, envelope(step.Apply), working, step.Manager, step.Force)
	}
	errorName := ""
	if err != nil {
		var conflicts merge.Conflicts
		if !errors.As(err, &conflicts) {
			return Snapshot{}, fmt.Errorf("unexpected managedfields error: %w", err)
		}
		errorName = "conflict"
	} else {
		if object != nil {
			e.live = object.(*unstructured.Unstructured)
		}
		e.fields = fields
	}
	// Exercise the actual FieldsV1 wire encoding between requests, rather than
	// assuming that internal field sets retain the same ownership after storage.
	wireObject := e.live.DeepCopy()
	if err := internal.EncodeObjectManagedFields(wireObject, e.fields); err != nil {
		return Snapshot{}, fmt.Errorf("encode managed fields: %w", err)
	}
	decoded, err := internal.DecodeManagedFields(wireObject.GetManagedFields())
	if err != nil {
		return Snapshot{}, fmt.Errorf("decode managed fields: %w", err)
	}
	if !decoded.Fields().Equals(e.fields.Fields()) {
		return Snapshot{}, fmt.Errorf("FieldsV1 round trip changed ownership, operation, or API version")
	}
	e.fields = decoded
	objectValue := runtime.DeepCopyJSON(e.live.Object)
	delete(objectValue, "apiVersion")
	delete(objectValue, "kind")
	flat, err := e.schema.flatten(objectValue, nil)
	cells := flat.cells
	if err != nil {
		return Snapshot{}, err
	}
	mapped, err := cellMap(cells)
	if err != nil {
		return Snapshot{}, err
	}
	out := Snapshot{errorName, mapped, map[string][]string{}}
	for identifier, versioned := range e.fields.Fields() {
		var entry metav1.ManagedFieldsEntry
		if err := json.Unmarshal([]byte(identifier), &entry); err != nil {
			return out, fmt.Errorf("invalid managedfields manager identity: %w", err)
		}
		if entry.Manager == "" || (entry.Operation != metav1.ManagedFieldsOperationApply && entry.Operation != metav1.ManagedFieldsOperationUpdate) || entry.Subresource != "" {
			return out, fmt.Errorf("unrepresentable manager identity %s", identifier)
		}
		operation := "apply"
		if entry.Operation == metav1.ManagedFieldsOperationUpdate {
			operation = "update"
		}
		manager := managerIdentity(entry.Manager, operation)
		if versioned.Applied() != (entry.Operation == metav1.ManagedFieldsOperationApply) || string(versioned.APIVersion()) != groupVersion.String() {
			return out, fmt.Errorf("unexpected manager version/operation for %q", manager)
		}
		var invalid error
		versioned.Set().Iterate(func(p fieldpath.Path) {
			if len(p) == 0 {
				invalid = fmt.Errorf("unexpected root ownership claim for %q", manager)
				return
			}
			if p[0].FieldName != nil && (*p[0].FieldName == "apiVersion" || *p[0].FieldName == "kind") {
				return
			}
			// Lean encodes a list-map key in membership, rather than as another leaf.
			if len(p) > 1 && p[len(p)-2].Key != nil && p[len(p)-1].FieldName != nil &&
				isKeyField(p) {
				return
			}
			key := p.String()
			_, exists := mapped[key]
			if !exists {
				invalid = fmt.Errorf("dangling or unrepresentable SMD claim %s by %q", key, manager)
				return
			}
			out.Owners[manager] = append(out.Owners[manager], key)
		})
		if invalid != nil {
			return out, invalid
		}
		if len(out.Owners[manager]) > 0 {
			sort.Strings(out.Owners[manager])
		}
	}
	return out, nil
}

// CheckGo runs every hand-authored expectation without requiring Lean.
func CheckGo(cases []Case) error {
	for _, c := range cases {
		engine, err := newGoEngine(c)
		if err != nil {
			return err
		}
		for i, step := range c.Steps {
			actual, err := engine.step(step)
			if err != nil {
				return fmt.Errorf("%s step %d: %w", c.Name, i+1, err)
			}
			expected := step.Expect
			if step.GoExpect != nil {
				expected = step.GoExpect
			}
			if expected != nil {
				want, err := expectedSnapshotFor(*expected, c)
				if err != nil {
					return err
				}
				if err := equalSnapshot(want, actual); err != nil {
					return fmt.Errorf("%s step %d Go oracle: %w", c.Name, i+1, err)
				}
			}
		}
	}
	return nil
}

// Key leaves repeat the identity already carried by the item selector.
func isKeyField(p fieldpath.Path) bool {
	if len(p) < 2 || p[len(p)-2].Key == nil || p[len(p)-1].FieldName == nil {
		return false
	}
	for _, key := range *p[len(p)-2].Key {
		if key.Name == *p[len(p)-1].FieldName {
			return true
		}
	}
	return false
}
