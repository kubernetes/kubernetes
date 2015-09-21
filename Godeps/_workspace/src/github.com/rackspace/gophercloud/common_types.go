package gophercloud

// Link is used for JSON (un)marshalling.
// It provides RESTful links to a resource.
type Link struct {
	Href string `json:"href"`
	Rel  string `json:"rel"`
	Type string `json:"type"`
}

// FileConfig structures represent a blob of data which must appear at a
// a specific location in a server's filesystem.  The file contents are
// base-64 encoded.
type FileConfig struct {
	Path     string `json:"path"`
	Contents string `json:"contents"`
}

// NetworkConfig structures represent an affinity between a server and a
// specific, uniquely identified network.  Networks are identified through
// universally unique IDs.
type NetworkConfig struct {
	Uuid string `json:"uuid"`
}
