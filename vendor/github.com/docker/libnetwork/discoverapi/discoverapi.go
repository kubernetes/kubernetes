package discoverapi

// Discover is an interface to be implemented by the component interested in receiving discover events
// like new node joining the cluster or datastore updates
type Discover interface {
	// DiscoverNew is a notification for a new discovery event, Example:a new node joining a cluster
	DiscoverNew(dType DiscoveryType, data interface{}) error

	// DiscoverDelete is a notification for a discovery delete event, Example:a node leaving a cluster
	DiscoverDelete(dType DiscoveryType, data interface{}) error
}

// DiscoveryType represents the type of discovery element the DiscoverNew function is invoked on
type DiscoveryType int

const (
	// NodeDiscovery represents Node join/leave events provided by discovery
	NodeDiscovery = iota + 1
	// DatastoreConfig represents an add/remove datastore event
	DatastoreConfig
	// EncryptionKeysConfig represents the initial key(s) for performing datapath encryption
	EncryptionKeysConfig
	// EncryptionKeysUpdate represents an update to the datapath encryption key(s)
	EncryptionKeysUpdate
)

// NodeDiscoveryData represents the structure backing the node discovery data json string
type NodeDiscoveryData struct {
	Address     string
	BindAddress string
	Self        bool
}

// DatastoreConfigData is the data for the datastore update event message
type DatastoreConfigData struct {
	Scope    string
	Provider string
	Address  string
	Config   interface{}
}

// DriverEncryptionConfig contains the initial datapath encryption key(s)
// Key in first position is the primary key, the one to be used in tx.
// Original key and tag types are []byte and uint64
type DriverEncryptionConfig struct {
	Keys [][]byte
	Tags []uint64
}

// DriverEncryptionUpdate carries an update to the encryption key(s) as:
// a new key and/or set a primary key and/or a removal of an existing key.
// Original key and tag types are []byte and uint64
type DriverEncryptionUpdate struct {
	Key        []byte
	Tag        uint64
	Primary    []byte
	PrimaryTag uint64
	Prune      []byte
	PruneTag   uint64
}
