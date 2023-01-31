package metadata

// ClientInfo wraps immutable data from the client.Client structure.
type ClientInfo struct {
	ServiceName    string
	ServiceID      string
	APIVersion     string
	PartitionID    string
	Endpoint       string
	SigningName    string
	SigningRegion  string
	JSONVersion    string
	TargetPrefix   string
	ResolvedRegion string
}
