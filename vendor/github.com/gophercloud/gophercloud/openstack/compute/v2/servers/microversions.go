package servers

// ExtractTags will extract the tags of a server.
// This requires the client to be set to microversion 2.26 or later.
func (r serverResult) ExtractTags() ([]string, error) {
	var s struct {
		Tags []string `json:"tags"`
	}
	err := r.ExtractInto(&s)
	return s.Tags, err
}
