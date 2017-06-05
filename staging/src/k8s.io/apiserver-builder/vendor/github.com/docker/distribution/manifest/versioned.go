package manifest

// Versioned provides a struct with the manifest schemaVersion and . Incoming
// content with unknown schema version can be decoded against this struct to
// check the version.
type Versioned struct {
	// SchemaVersion is the image manifest schema that this image follows
	SchemaVersion int `json:"schemaVersion"`

	// MediaType is the media type of this schema.
	MediaType string `json:"mediaType,omitempty"`
}
