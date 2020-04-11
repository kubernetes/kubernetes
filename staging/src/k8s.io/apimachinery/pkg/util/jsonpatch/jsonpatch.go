package jsonpatch

import "github.com/evanphx/json-patch"

func init() {
	// Disable negative indices to be compliant with RFC6902.
	jsonpatch.SupportNegativeIndices = false
}

// CreateMergePatch delegates to jsonpatch.CreateMergePatch
func CreateMergePatch(originalJSON, modifiedJSON []byte) ([]byte, error) {
	return jsonpatch.CreateMergePatch(originalJSON, modifiedJSON)
}

// MergePatch delegates to jsonpatch.CreateMergePatch
func MergePatch(docData, patchData []byte) ([]byte, error) {
	return jsonpatch.MergePatch(docData, patchData)
}

// MergeMergePatches delegates to jsonpatch.MergeMergePatches
func MergeMergePatches(patch1Data, patch2Data []byte) ([]byte, error) {
	return jsonpatch.MergeMergePatches(patch1Data, patch2Data)
}

// Equal delegates to jsonpatch.Equal
func Equal(a, b []byte) bool {
	return jsonpatch.Equal(a, b)
}

// DecodePatch delegates to jsonpatch.DecodePatch
func DecodePatch(buf []byte) (jsonpatch.Patch, error) {
	return jsonpatch.DecodePatch(buf)
}
