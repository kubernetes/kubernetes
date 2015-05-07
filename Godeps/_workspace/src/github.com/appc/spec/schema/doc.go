// Package schema provides definitions for the JSON schema of the different
// manifests in the App Container Specification. The manifests are canonically
// represented in their respective structs:
//   - `ImageManifest`
//   - `PodManifest`
//
// Validation is performed through serialization: if a blob of JSON data will
// unmarshal to one of the *Manifests, it is considered a valid implementation
// of the standard. Similarly, if a constructed *Manifest struct marshals
// successfully to JSON, it must be valid.
package schema
