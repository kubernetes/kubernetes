package availabilityzones

// ServerAvailabilityZoneExt is an extension to the base Server result which
// includes the Availability Zone information.
type ServerAvailabilityZoneExt struct {
	// AvailabilityZone is the availabilty zone the server is in.
	AvailabilityZone string `json:"OS-EXT-AZ:availability_zone"`
}
