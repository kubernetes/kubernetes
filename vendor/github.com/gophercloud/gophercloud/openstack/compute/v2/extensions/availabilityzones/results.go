package availabilityzones

// ServerExt is an extension to the base Server object
type ServerExt struct {
	// AvailabilityZone is the availabilty zone the server is in.
	AvailabilityZone string `json:"OS-EXT-AZ:availability_zone"`
}

// UnmarshalJSON to override default
func (r *ServerExt) UnmarshalJSON(b []byte) error {
	return nil
}
