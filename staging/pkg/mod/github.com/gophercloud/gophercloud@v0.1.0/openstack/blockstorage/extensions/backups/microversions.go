package backups

// ExtractMetadata will extract the metadata of a backup.
// This requires the client to be set to microversion 3.43 or later.
func (r commonResult) ExtractMetadata() (map[string]string, error) {
	var s struct {
		Metadata map[string]string `json:"metadata"`
	}
	err := r.ExtractInto(&s)
	return s.Metadata, err
}

// ExtractAvailaiblityZone will extract the availability zone of a backup.
// This requires the client to be set to microversion 3.51 or later.
func (r commonResult) ExtractAvailabilityZone() (string, error) {
	var s struct {
		AvailabilityZone string `json:"availability_zone"`
	}
	err := r.ExtractInto(&s)
	return s.AvailabilityZone, err
}
