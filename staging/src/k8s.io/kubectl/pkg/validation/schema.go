/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"bytes"
	"encoding/json"
	"fmt"

	ejson "github.com/exponent-io/jsonpath"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/klog/v2"
)

// Schema is an interface that knows how to validate an API object serialized to a byte array.
type Schema interface {
	ValidateBytes(data []byte) error
}

// NullSchema always validates bytes.
type NullSchema struct{}

// ValidateBytes never fails for NullSchema.
func (NullSchema) ValidateBytes(data []byte) error { return nil }

// NoDoubleKeySchema is a schema that disallows double keys.
type NoDoubleKeySchema struct{}

// ValidateBytes validates bytes.
func (NoDoubleKeySchema) ValidateBytes(data []byte) error {
	var list []error
	if err := validateNoDuplicateKeys(data, "metadata", "labels"); err != nil {
		list = append(list, err)
	}
	if err := validateNoDuplicateKeys(data, "metadata", "annotations"); err != nil {
		list = append(list, err)
	}
	return utilerrors.NewAggregate(list)
}

func validateNoDuplicateKeys(data []byte, path ...string) error {
	r := ejson.NewDecoder(bytes.NewReader(data))
	// This is Go being unfriendly. The 'path ...string' comes in as a
	// []string, and SeekTo takes ...interface{}, so we can't just pass
	// the path straight in, we have to copy it.  *sigh*
	ifacePath := []interface{}{}
	for ix := range path {
		ifacePath = append(ifacePath, path[ix])
	}
	found, err := r.SeekTo(ifacePath...)
	if err != nil {
		return err
	}
	if !found {
		return nil
	}
	seen := map[string]bool{}
	for {
		tok, err := r.Token()
		if err != nil {
			return err
		}
		switch t := tok.(type) {
		case json.Delim:
			if t.String() == "}" {
				return nil
			}
		case ejson.KeyString:
			if seen[string(t)] {
				return fmt.Errorf("duplicate key: %s", string(t))
			}
			seen[string(t)] = true
		}
	}
}

// ConjunctiveSchema encapsulates a schema list.
type ConjunctiveSchema []Schema

// ValidateBytes validates bytes per a ConjunctiveSchema.
func (c ConjunctiveSchema) ValidateBytes(data []byte) error {
	var list []error
	schemas := []Schema(c)
	for ix := range schemas {
		if err := schemas[ix].ValidateBytes(data); err != nil {
			list = append(list, err)
		}
	}
	return utilerrors.NewAggregate(list)
}

func NewParamVerifyingSchema(s Schema, verifier resource.Verifier, directive string) Schema {
	return &paramVerifyingSchema{
		schema:    s,
		verifier:  verifier,
		directive: directive,
	}
}

// paramVerifyingSchema only performs validation
// based on the fieldValidation query param
// being unsupported by the apiserver, because
// server-side validation will be performed instead
// of client-side validation.
type paramVerifyingSchema struct {
	schema    Schema
	verifier  resource.Verifier
	directive string
}

// ValidateBytes validates bytes per a ParamVerifyingSchema
func (c *paramVerifyingSchema) ValidateBytes(data []byte) error {
	obj, err := parse(data)
	if err != nil {
		return err
	}

	gvk, errs := getObjectKind(obj)
	if errs != nil {
		return utilerrors.NewAggregate(errs)
	}

	err = c.verifier.HasSupport(gvk)
	if resource.IsParamUnsupportedError(err) {
		switch c.directive {
		case metav1.FieldValidationStrict:
			return c.schema.ValidateBytes(data)
		case metav1.FieldValidationWarn:
			klog.Warningf("cannot perform warn validation if server-side field validation is unsupported, skipping validation")
		default:
			// can't be reached
			klog.Warningf("unexpected field validation directive: %s, skipping validation", c.directive)
		}
		return nil
	}
	return err
}
