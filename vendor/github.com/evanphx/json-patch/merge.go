package jsonpatch

import (
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

// CreateMergePatch creates a merge patch as specified in http://tools.ietf.org/html/draft-ietf-appsawg-json-merge-patch-07
//
// 'a' is original, 'b' is the modified document. Both are to be given as json encoded content.
// The function will return a mergeable json document with differences from a to b.
//
// An error will be returned if any of the two documents are invalid.
func CreateMergePatch(a, b []byte) ([]byte, error) {
	aI := map[string]interface{}{}
	bI := map[string]interface{}{}
	err := json.Unmarshal(a, &aI)
	if err != nil {
		return nil, errBadJSONDoc
	}
	err = json.Unmarshal(b, &bI)
	if err != nil {
		return nil, errBadJSONDoc
	}
	dest, err := getDiff(aI, bI)
	if err != nil {
		return nil, err
	}
	return json.Marshal(dest)
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
		for key := range at {
			if !matchesValue(at[key], bt[key]) {
				return false
			}
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
