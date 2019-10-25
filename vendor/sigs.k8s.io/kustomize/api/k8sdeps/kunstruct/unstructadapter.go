// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package kunstruct provides unstructured from api machinery and factory for creating unstructured
package kunstruct

import (
	"encoding/json"
	"fmt"

	jsonpatch "github.com/evanphx/json-patch"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/resid"
)

var _ ifc.Kunstructured = &UnstructAdapter{}

// UnstructAdapter wraps unstructured.Unstructured from
// https://github.com/kubernetes/apimachinery/blob/master/
//     pkg/apis/meta/v1/unstructured/unstructured.go
// to isolate dependence on apimachinery.
type UnstructAdapter struct {
	unstructured.Unstructured
}

// NewKunstructuredFromObject returns a new instance of Kunstructured.
func NewKunstructuredFromObject(obj runtime.Object) (ifc.Kunstructured, error) {
	// Convert obj to a byte stream, then convert that to JSON (Unstructured).
	marshaled, err := json.Marshal(obj)
	if err != nil {
		return &UnstructAdapter{}, err
	}
	var u unstructured.Unstructured
	err = u.UnmarshalJSON(marshaled)
	// creationTimestamp always 'null', remove it
	u.SetCreationTimestamp(metav1.Time{})
	return &UnstructAdapter{Unstructured: u}, err
}

// GetGvk returns the Gvk name of the object.
func (fs *UnstructAdapter) GetGvk() resid.Gvk {
	x := fs.GroupVersionKind()
	return resid.Gvk{
		Group:   x.Group,
		Version: x.Version,
		Kind:    x.Kind,
	}
}

// SetGvk set the Gvk of the object to the input Gvk
func (fs *UnstructAdapter) SetGvk(g resid.Gvk) {
	fs.SetGroupVersionKind(toSchemaGvk(g))
}

// Copy provides a copy behind an interface.
func (fs *UnstructAdapter) Copy() ifc.Kunstructured {
	return &UnstructAdapter{*fs.DeepCopy()}
}

// Map returns the unstructured content map.
func (fs *UnstructAdapter) Map() map[string]interface{} {
	return fs.Object
}

// SetMap overrides the unstructured content map.
func (fs *UnstructAdapter) SetMap(m map[string]interface{}) {
	fs.Object = m
}

func (fs *UnstructAdapter) selectSubtree(path string) (map[string]interface{}, []string, bool, error) {
	sections, err := parseFields(path)
	if len(sections) == 0 || (err != nil) {
		return nil, nil, false, err
	}

	content := fs.UnstructuredContent()
	lastSectionIdx := len(sections)

	// There are multiple sections to walk
	for sectionIdx := 0; sectionIdx < lastSectionIdx; sectionIdx++ {
		idx := sections[sectionIdx].idx
		fields := sections[sectionIdx].fields

		if idx == -1 {
			// This section has no index
			return content, fields, true, nil
		}

		// This section is terminated by an indexed field.
		// Let's extract the slice first
		indexedField, found, err := unstructured.NestedFieldNoCopy(content, fields...)
		if !found || err != nil {
			return content, fields, found, err
		}
		s, ok := indexedField.([]interface{})
		if !ok {
			return content, fields, false, fmt.Errorf("%v is of the type %T, expected []interface{}", indexedField, indexedField)
		}
		if idx >= len(s) {
			return content, fields, false, fmt.Errorf("index %d is out of bounds", idx)
		}

		if sectionIdx == lastSectionIdx-1 {
			// This is the last section. Let's build a fake map
			// to let the rest of the field extraction to work.
			idxstring := fmt.Sprintf("[%v]", idx)
			newContent := map[string]interface{}{idxstring: s[idx]}
			newFields := []string{idxstring}
			return newContent, newFields, true, nil
		}

		newContent, ok := s[idx].(map[string]interface{})
		if !ok {
			// Only map are supported here
			return content, fields, false,
				fmt.Errorf("%#v is expected to be of type map[string]interface{}", s[idx])
		}
		content = newContent
	}

	// It seems to be an invalid path
	return nil, []string{}, false, nil
}

// GetFieldValue returns the value at the given fieldpath.
func (fs *UnstructAdapter) GetFieldValue(path string) (interface{}, error) {
	content, fields, found, err := fs.selectSubtree(path)
	if !found || err != nil {
		return nil, noFieldError{Field: path}
	}

	s, found, err := unstructured.NestedFieldNoCopy(
		content, fields...)
	if found || err != nil {
		return s, err
	}
	return nil, noFieldError{Field: path}
}

// GetString returns value at the given fieldpath.
func (fs *UnstructAdapter) GetString(path string) (string, error) {
	content, fields, found, err := fs.selectSubtree(path)
	if !found || err != nil {
		return "", noFieldError{Field: path}
	}

	s, found, err := unstructured.NestedString(
		content, fields...)
	if found || err != nil {
		return s, err
	}
	return "", noFieldError{Field: path}
}

// GetStringSlice returns value at the given fieldpath.
func (fs *UnstructAdapter) GetStringSlice(path string) ([]string, error) {
	content, fields, found, err := fs.selectSubtree(path)
	if !found || err != nil {
		return []string{}, noFieldError{Field: path}
	}

	s, found, err := unstructured.NestedStringSlice(
		content, fields...)
	if found || err != nil {
		return s, err
	}
	return []string{}, noFieldError{Field: path}
}

// GetBool returns value at the given fieldpath.
func (fs *UnstructAdapter) GetBool(path string) (bool, error) {
	content, fields, found, err := fs.selectSubtree(path)
	if !found || err != nil {
		return false, noFieldError{Field: path}
	}

	s, found, err := unstructured.NestedBool(
		content, fields...)
	if found || err != nil {
		return s, err
	}
	return false, noFieldError{Field: path}
}

// GetFloat64 returns value at the given fieldpath.
func (fs *UnstructAdapter) GetFloat64(path string) (float64, error) {
	content, fields, found, err := fs.selectSubtree(path)
	if !found || err != nil {
		return 0, err
	}

	s, found, err := unstructured.NestedFloat64(
		content, fields...)
	if found || err != nil {
		return s, err
	}
	return 0, noFieldError{Field: path}
}

// GetInt64 returns value at the given fieldpath.
func (fs *UnstructAdapter) GetInt64(path string) (int64, error) {
	content, fields, found, err := fs.selectSubtree(path)
	if !found || err != nil {
		return 0, noFieldError{Field: path}
	}

	s, found, err := unstructured.NestedInt64(
		content, fields...)
	if found || err != nil {
		return s, err
	}
	return 0, noFieldError{Field: path}
}

// GetSlice returns value at the given fieldpath.
func (fs *UnstructAdapter) GetSlice(path string) ([]interface{}, error) {
	content, fields, found, err := fs.selectSubtree(path)
	if !found || err != nil {
		return nil, noFieldError{Field: path}
	}

	s, found, err := unstructured.NestedSlice(
		content, fields...)
	if found || err != nil {
		return s, err
	}
	return nil, noFieldError{Field: path}
}

// GetStringMap returns value at the given fieldpath.
func (fs *UnstructAdapter) GetStringMap(path string) (map[string]string, error) {
	content, fields, found, err := fs.selectSubtree(path)
	if !found || err != nil {
		return nil, noFieldError{Field: path}
	}

	s, found, err := unstructured.NestedStringMap(
		content, fields...)
	if found || err != nil {
		return s, err
	}
	return nil, noFieldError{Field: path}
}

// GetMap returns value at the given fieldpath.
func (fs *UnstructAdapter) GetMap(path string) (map[string]interface{}, error) {
	content, fields, found, err := fs.selectSubtree(path)
	if !found || err != nil {
		return nil, noFieldError{Field: path}
	}

	s, found, err := unstructured.NestedMap(
		content, fields...)
	if found || err != nil {
		return s, err
	}
	return nil, noFieldError{Field: path}
}

func (fs *UnstructAdapter) MatchesLabelSelector(selector string) (bool, error) {
	s, err := labels.Parse(selector)
	if err != nil {
		return false, err
	}
	return s.Matches(labels.Set(fs.GetLabels())), nil
}

func (fs *UnstructAdapter) MatchesAnnotationSelector(selector string) (bool, error) {
	s, err := labels.Parse(selector)
	if err != nil {
		return false, err
	}
	return s.Matches(labels.Set(fs.GetAnnotations())), nil
}

func (fs *UnstructAdapter) Patch(patch ifc.Kunstructured) error {
	versionedObj, err := scheme.Scheme.New(
		toSchemaGvk(patch.GetGvk()))
	merged := map[string]interface{}{}
	saveName := fs.GetName()
	switch {
	case runtime.IsNotRegisteredError(err):
		baseBytes, err := json.Marshal(fs.Map())
		if err != nil {
			return err
		}
		patchBytes, err := json.Marshal(patch.Map())
		if err != nil {
			return err
		}
		mergedBytes, err := jsonpatch.MergePatch(baseBytes, patchBytes)
		if err != nil {
			return err
		}
		err = json.Unmarshal(mergedBytes, &merged)
		if err != nil {
			return err
		}
	case err != nil:
		return err
	default:
		// Use Strategic-Merge-Patch to handle types w/ schema
		// TODO: Change this to use the new Merge package.
		// Store the name of the target object, because this name may have been munged.
		// Apply this name to the patched object.
		lookupPatchMeta, err := strategicpatch.NewPatchMetaFromStruct(versionedObj)
		if err != nil {
			return err
		}
		merged, err = strategicpatch.StrategicMergeMapPatchUsingLookupPatchMeta(
			fs.Map(),
			patch.Map(),
			lookupPatchMeta)
		if err != nil {
			return err
		}
	}
	fs.SetMap(merged)
	if len(fs.Map()) != 0 {
		// if the patch deletes the object
		// don't reset the name
		fs.SetName(saveName)
	}
	return nil
}

// toSchemaGvk converts to a schema.GroupVersionKind.
func toSchemaGvk(x resid.Gvk) schema.GroupVersionKind {
	return schema.GroupVersionKind{
		Group:   x.Group,
		Version: x.Version,
		Kind:    x.Kind,
	}
}

// noFieldError is returned when a field is expected, but missing.
type noFieldError struct {
	Field string
}

func (e noFieldError) Error() string {
	return fmt.Sprintf("no field named '%s'", e.Field)
}
