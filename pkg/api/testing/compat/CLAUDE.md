# Package: compat

## Purpose
Provides compatibility testing utilities for verifying API object serialization across versions, ensuring that expected fields are present and absent fields remain absent after round-trip encoding.

## Key Functions
- `TestCompatibility(t *testing.T, version schema.GroupVersion, input []byte, validator func(obj runtime.Object) field.ErrorList, expectedKeys map[string]string, absentKeys []string)` - Tests that:
  1. Input JSON decodes successfully to the given version
  2. Object passes validation
  3. Re-encoded JSON contains all expected keys with expected values
  4. Re-encoded JSON does not contain any of the absent keys

### Helper Function
- `getJSONValue(data map[string]interface{}, keys ...string) (interface{}, bool, error)` - Traverses a JSON object using dot-separated keys, supports array indexing with `[n]` syntax (e.g., `spec.containers[0].name`)

## Usage Pattern
```go
TestCompatibility(t, v1.SchemeGroupVersion,
    []byte(`{"apiVersion":"v1","kind":"Pod",...}`),
    validatePod,
    map[string]string{"spec.containers[0].name": "mycontainer"},
    []string{"spec.deprecatedField"},
)
```

## Design Notes
- Based on OpenShift's compatibility testing pattern
- Useful for verifying that deprecated fields are properly removed during serialization
- Helps ensure API stability across Kubernetes versions
