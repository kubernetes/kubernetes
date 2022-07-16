// Copyright 2015 xeipuuv ( https://github.com/xeipuuv )
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// author           xeipuuv
// author-github    https://github.com/xeipuuv
// author-mail      xeipuuv@gmail.com
//
// repository-name  gojsonschema
// repository-desc  An implementation of JSON Schema, based on IETF's draft v4 - Go language.
//
// description		Defines resources pooling.
//                  Eases referencing and avoids downloading the same resource twice.
//
// created          26-02-2013

package gojsonschema

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/xeipuuv/gojsonreference"
)

type schemaPoolDocument struct {
	Document interface{}
	Draft    *Draft
}

type schemaPool struct {
	schemaPoolDocuments map[string]*schemaPoolDocument
	jsonLoaderFactory   JSONLoaderFactory
	autoDetect          *bool
}

func (p *schemaPool) parseReferences(document interface{}, ref gojsonreference.JsonReference, pooled bool) error {

	var (
		draft     *Draft
		err       error
		reference = ref.String()
	)
	// Only the root document should be added to the schema pool if pooled is true
	if _, ok := p.schemaPoolDocuments[reference]; pooled && ok {
		return fmt.Errorf("Reference already exists: \"%s\"", reference)
	}

	if *p.autoDetect {
		_, draft, err = parseSchemaURL(document)
		if err != nil {
			return err
		}
	}

	err = p.parseReferencesRecursive(document, ref, draft)

	if pooled {
		p.schemaPoolDocuments[reference] = &schemaPoolDocument{Document: document, Draft: draft}
	}

	return err
}

func (p *schemaPool) parseReferencesRecursive(document interface{}, ref gojsonreference.JsonReference, draft *Draft) error {
	// parseReferencesRecursive parses a JSON document and resolves all $id and $ref references.
	// For $ref references it takes into account the $id scope it is in and replaces
	// the reference by the absolute resolved reference

	// When encountering errors it fails silently. Error handling is done when the schema
	// is syntactically parsed and any error encountered here should also come up there.
	switch m := document.(type) {
	case []interface{}:
		for _, v := range m {
			p.parseReferencesRecursive(v, ref, draft)
		}
	case map[string]interface{}:
		localRef := &ref

		keyID := KEY_ID_NEW
		if existsMapKey(m, KEY_ID) {
			keyID = KEY_ID
		}
		if existsMapKey(m, keyID) && isKind(m[keyID], reflect.String) {
			jsonReference, err := gojsonreference.NewJsonReference(m[keyID].(string))
			if err == nil {
				localRef, err = ref.Inherits(jsonReference)
				if err == nil {
					if _, ok := p.schemaPoolDocuments[localRef.String()]; ok {
						return fmt.Errorf("Reference already exists: \"%s\"", localRef.String())
					}
					p.schemaPoolDocuments[localRef.String()] = &schemaPoolDocument{Document: document, Draft: draft}
				}
			}
		}

		if existsMapKey(m, KEY_REF) && isKind(m[KEY_REF], reflect.String) {
			jsonReference, err := gojsonreference.NewJsonReference(m[KEY_REF].(string))
			if err == nil {
				absoluteRef, err := localRef.Inherits(jsonReference)
				if err == nil {
					m[KEY_REF] = absoluteRef.String()
				}
			}
		}

		for k, v := range m {
			// const and enums should be interpreted literally, so ignore them
			if k == KEY_CONST || k == KEY_ENUM {
				continue
			}
			// Something like a property or a dependency is not a valid schema, as it might describe properties named "$ref", "$id" or "const", etc
			// Therefore don't treat it like a schema.
			if k == KEY_PROPERTIES || k == KEY_DEPENDENCIES || k == KEY_PATTERN_PROPERTIES {
				if child, ok := v.(map[string]interface{}); ok {
					for _, v := range child {
						p.parseReferencesRecursive(v, *localRef, draft)
					}
				}
			} else {
				p.parseReferencesRecursive(v, *localRef, draft)
			}
		}
	}
	return nil
}

func (p *schemaPool) GetDocument(reference gojsonreference.JsonReference) (*schemaPoolDocument, error) {

	var (
		spd   *schemaPoolDocument
		draft *Draft
		ok    bool
		err   error
	)

	if internalLogEnabled {
		internalLog("Get Document ( %s )", reference.String())
	}

	// Create a deep copy, so we can remove the fragment part later on without altering the original
	refToURL, _ := gojsonreference.NewJsonReference(reference.String())

	// First check if the given fragment is a location independent identifier
	// http://json-schema.org/latest/json-schema-core.html#rfc.section.8.2.3

	if spd, ok = p.schemaPoolDocuments[refToURL.String()]; ok {
		if internalLogEnabled {
			internalLog(" From pool")
		}
		return spd, nil
	}

	// If the given reference is not a location independent identifier,
	// strip the fragment and look for a document with it's base URI

	refToURL.GetUrl().Fragment = ""

	if cachedSpd, ok := p.schemaPoolDocuments[refToURL.String()]; ok {
		document, _, err := reference.GetPointer().Get(cachedSpd.Document)

		if err != nil {
			return nil, err
		}

		if internalLogEnabled {
			internalLog(" From pool")
		}

		spd = &schemaPoolDocument{Document: document, Draft: cachedSpd.Draft}
		p.schemaPoolDocuments[reference.String()] = spd

		return spd, nil
	}

	// It is not possible to load anything remotely that is not canonical...
	if !reference.IsCanonical() {
		return nil, errors.New(formatErrorDescription(
			Locale.ReferenceMustBeCanonical(),
			ErrorDetails{"reference": reference.String()},
		))
	}

	jsonReferenceLoader := p.jsonLoaderFactory.New(reference.String())
	document, err := jsonReferenceLoader.LoadJSON()

	if err != nil {
		return nil, err
	}

	// add the whole document to the pool for potential re-use
	p.parseReferences(document, refToURL, true)

	_, draft, _ = parseSchemaURL(document)

	// resolve the potential fragment and also cache it
	document, _, err = reference.GetPointer().Get(document)

	if err != nil {
		return nil, err
	}

	return &schemaPoolDocument{Document: document, Draft: draft}, nil
}
