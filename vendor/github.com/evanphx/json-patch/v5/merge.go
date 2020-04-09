package jsonpatch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
)

func merge(cur, patch *lazyNode, mergeMerge bool) *lazyNode {
	curDoc, err := cur.intoDoc()

	if err != nil {
		pruneNulls(patch)
		return patch
	}

	patchDoc, err := patch.intoDoc()

	if err != nil {
		return patch
	}

	mergeDocs(curDoc, patchDoc, mergeMerge)

	return cur
}

func mergeDocs(doc, patch *partialDoc, mergeMerge bool) {
	for k, v := range *patch {
		if v == nil {
			if mergeMerge {
				(*doc)[k] = nil
			} else {
				delete(*doc, k)
			}
		} else {
			cur, ok := (*doc)[k]

			if !ok || cur == nil {
				pruneNulls(v)
				(*doc)[k] = v
			} else {
				(*doc)[k] = merge(cur, v, mergeMerge)
			}
		}
	}
}

func pruneNulls(n *lazyNode) {
	sub, err := n.intoDoc()

	if err == nil {
		pruneDocNulls(sub)
	} else {
		ary, err := n.intoAry()

		if err == nil {
			pruneAryNulls(ary)
		}
	}
}

func pruneDocNulls(doc *partialDoc) *partialDoc {
	for k, v := range *doc {
		if v == nil {
			delete(*doc, k)
		} else {
			pruneNulls(v)
		}
	}

	return doc
}

func pruneAryNulls(ary *partialArray) *partialArray {
	newAry := []*lazyNode{}

	for _, v := range *ary {
		if v != nil {
			pruneNulls(v)
			newAry = append(newAry, v)
		}
	}

	*ary = newAry

	return ary
}

var errBadJSONDoc = fmt.Errorf("Invalid JSON Document")
var errBadJSONPatch = fmt.Errorf("Invalid JSON Patch")
var errBadMergeTypes = fmt.Errorf("Mismatched JSON Documents")

// MergeMergePatches merges two merge patches together, such that
// applying this resulting merged merge patch to a document yields the same
// as merging each merge patch to the document in succession.
func MergeMergePatches(patch1Data, patch2Data []byte) ([]byte, error) {
	return doMergePatch(patch1Data, patch2Data, true)
}

// MergePatch merges the patchData into the docData.
func MergePatch(docData, patchData []byte) ([]byte, error) {
	return doMergePatch(docData, patchData, false)
}

func doMergePatch(docData, patchData []byte, mergeMerge bool) ([]byte, error) {
	doc := &partialDoc{}

	docErr := json.Unmarshal(docData, doc)

	patch := &partialDoc{}

	patchErr := json.Unmarshal(patchData, patch)

	if _, ok := docErr.(*json.SyntaxError); ok {
		return nil, errBadJSONDoc
	}

	if _, ok := patchErr.(*json.SyntaxError); ok {
		return nil, errBadJSONPatch
	}

	if docErr == nil && *doc == nil {
		return nil, errBadJSONDoc
	}

	if patchErr == nil && *patch == nil {
		return nil, errBadJSONPatch
	}

	if docErr != nil || patchErr != nil {
		// Not an error, just not a doc, so we turn straight into the patch
		if patchErr == nil {
			if mergeMerge {
				doc = patch
			} else {
				doc = pruneDocNulls(patch)
			}
		} else {
			patchAry := &partialArray{}
			patchErr = json.Unmarshal(patchData, patchAry)

			if patchErr != nil {
				return nil, errBadJSONPatch
			}

			pruneAryNulls(patchAry)

			out, patchErr := json.Marshal(patchAry)

			if patchErr != nil {
				return nil, errBadJSONPatch
			}

			return out, nil
		}
	} else {
		mergeDocs(doc, patch, mergeMerge)
	}

	return json.Marshal(doc)
}

// resemblesJSONArray indicates whether the byte-slice "appears" to be
// a JSON array or not.
// False-positives are possible, as this function does not check the internal
// structure of the array. It only checks that the outer syntax is present and
// correct.
func resemblesJSONArray(input []byte) bool {
	input = bytes.TrimSpace(input)

	hasPrefix := bytes.HasPrefix(input, []byte("["))
	hasSuffix := bytes.HasSuffix(input, []byte("]"))

	return hasPrefix && hasSuffix
}

// CreateMergePatch will return a merge patch document capable of converting
// the original document(s) to the modified document(s).
// The parameters can be bytes of either two JSON Documents, or two arrays of
// JSON documents.
// The merge patch returned follows the specification defined at http://tools.ietf.org/html/draft-ietf-appsawg-json-merge-patch-07
func CreateMergePatch(originalJSON, modifiedJSON []byte) ([]byte, error) {
	originalResemblesArray := resemblesJSONArray(originalJSON)
	modifiedResemblesArray := resemblesJSONArray(modifiedJSON)

	// Do both byte-slices seem like JSON arrays?
	if originalResemblesArray && modifiedResemblesArray {
		return createArrayMergePatch(originalJSON, modifiedJSON)
	}

	// Are both byte-slices are not arrays? Then they are likely JSON objects...
	if !originalResemblesArray && !modifiedResemblesArray {
		return createObjectMergePatch(originalJSON, modifiedJSON)
	}

	// None of the above? Then return an error because of mismatched types.
	return nil, errBadMergeTypes
}

// createObjectMergePatch will return a merge-patch document capable of
// converting the original document to the modified document.
func createObjectMergePatch(originalJSON, modifiedJSON []byte) ([]byte, error) {
	originalDoc := map[string]interface{}{}
	modifiedDoc := map[string]interface{}{}

	err := json.Unmarshal(originalJSON, &originalDoc)
	if err != nil {
		return nil, errBadJSONDoc
	}

	err = json.Unmarshal(modifiedJSON, &modifiedDoc)
	if err != nil {
		return nil, errBadJSONDoc
	}

	dest, err := getDiff(originalDoc, modifiedDoc)
	if err != nil {
		return nil, err
	}

	return json.Marshal(dest)
}

// createArrayMergePatch will return an array of merge-patch documents capable
// of converting the original document to the modified document for each
// pair of JSON documents provided in the arrays.
// Arrays of mismatched sizes will result in an error.
func createArrayMergePatch(originalJSON, modifiedJSON []byte) ([]byte, error) {
	originalDocs := []json.RawMessage{}
	modifiedDocs := []json.RawMessage{}

	err := json.Unmarshal(originalJSON, &originalDocs)
	if err != nil {
		return nil, errBadJSONDoc
	}

	err = json.Unmarshal(modifiedJSON, &modifiedDocs)
	if err != nil {
		return nil, errBadJSONDoc
	}

	total := len(originalDocs)
	if len(modifiedDocs) != total {
		return nil, errBadJSONDoc
	}

	result := []json.RawMessage{}
	for i := 0; i < len(originalDocs); i++ {
		original := originalDocs[i]
		modified := modifiedDocs[i]

		patch, err := createObjectMergePatch(original, modified)
		if err != nil {
			return nil, err
		}

		result = append(result, json.RawMessage(patch))
	}

	return json.Marshal(result)
}

// Returns true if the array matches (must be json types).
// As is idiomatic for go, an empty array is not the same as a nil array.
func matchesArray(a, b []interface{}) bool {
	if len(a) != len(b) {
		return false
	}
	if (a == nil && b != nil) || (a != nil && b == nil) {
		return false
	}
	for i := range a {
		if !matchesValue(a[i], b[i]) {
			return false
		}
	}
	return true
}

// Returns true if the values matches (must be json types)
// The types of the values must match, otherwise it will always return false
// If two map[string]interface{} are given, all elements must match.
func matchesValue(av, bv interface{}) bool {
	if reflect.TypeOf(av) != reflect.TypeOf(bv) {
		return false
	}
	switch at := av.(type) {
	case string:
		bt := bv.(string)
		if bt == at {
			return true
		}
	case float64:
		bt := bv.(float64)
		if bt == at {
			return true
		}
	case bool:
		bt := bv.(bool)
		if bt == at {
			return true
		}
	case nil:
		// Both nil, fine.
		return true
	case map[string]interface{}:
		bt := bv.(map[string]interface{})
		if len(bt) != len(at) {
			return false
		}
		for key := range bt {
			if !matchesValue(at[key], bt[key]) {
				return false
			}
		}
		return true
	case []interface{}:
		bt := bv.([]interface{})
		return matchesArray(at, bt)
	}
	return false
}

// getDiff returns the (recursive) difference between a and b as a map[string]interface{}.
func getDiff(a, b map[string]interface{}) (map[string]interface{}, error) {
	into := map[string]interface{}{}
	for key, bv := range b {
		av, ok := a[key]
		// value was added
		if !ok {
			into[key] = bv
			continue
		}
		// If types have changed, replace completely
		if reflect.TypeOf(av) != reflect.TypeOf(bv) {
			into[key] = bv
			continue
		}
		// Types are the same, compare values
		switch at := av.(type) {
		case map[string]interface{}:
			bt := bv.(map[string]interface{})
			dst := make(map[string]interface{}, len(bt))
			dst, err := getDiff(at, bt)
			if err != nil {
				return nil, err
			}
			if len(dst) > 0 {
				into[key] = dst
			}
		case string, float64, bool:
			if !matchesValue(av, bv) {
				into[key] = bv
			}
		case []interface{}:
			bt := bv.([]interface{})
			if !matchesArray(at, bt) {
				into[key] = bv
			}
		case nil:
			switch bv.(type) {
			case nil:
				// Both nil, fine.
			default:
				into[key] = bv
			}
		default:
			panic(fmt.Sprintf("Unknown type:%T in key %s", av, key))
		}
	}
	// Now add all deleted values as nil
	for key := range a {
		_, found := b[key]
		if !found {
			into[key] = nil
		}
	}
	return into, nil
}
