package consul

import (
	"fmt"
	"io"
	"net"
	"os"
	"time"

	"github.com/hashicorp/consul/tlsutil"
	"github.com/hashicorp/memberlist"
	"github.com/hashicorp/raft"
	"github.com/hashicorp/serf/serf"
)

const (
	DefaultDC          = "dc1"
	DefaultLANSerfPort = 8301
	DefaultWANSerfPort = 8302
)

var (
	DefaultRPCAddr = &net.TCPAddr{IP: net.ParseIP("0.0.0.0"), Port: 8300}
)

// ProtocolVersionMap is the mapping of Consul protocol versions
// to Serf protocol versions. We mask the Serf protocols using
// our own protocol version.
var protocolVersionMap map[uint8]uint8

func init() {
	protocolVersionMap = map[uint8]uint8{
		1: 4,
		2: 4,
		3: 4,
	}
}

// Config is used to configure the server
type Config struct {
	// Bootstrap mode is used to bring up the first Consul server.
	// It is required so that it can elect a leader without any
	// other nodes being present
	Bootstrap bool

	// BootstrapExpect mode is used to automatically bring up a collection of
	// Consul servers. This can be used to automatically bring up a collection
	// of nodes.
	BootstrapExpect int

	// Datacenter is the datacenter this Consul server represents
	Datacenter string

	// DataDir is the directory to store our state in
	DataDir string

	// DevMode is used to enable a development server mode.
	DevMode bool

	// Node name is the name we use to advertise. Defaults to hostname.
	NodeName string

	// Domain is the DNS domain for the records. Defaults to "consul."
	Domain string

	// RaftConfig is the configuration used for Raft in the local DC
	RaftConfig *raft.Config

	// RPCAddr is the RPC address used by Consul. This should be reachable
	// by the WAN and LAN
	RPCAddr *net.TCPAddr

	// RPCAdvertise is the address that is advertised to other nodes for
	// the RPC endpoint. This can differ from the RPC address, if for example
	// the RPCAddr is unspecified "0.0.0.0:8300", but this address must be
	// reachable
	RPCAdvertise *net.TCPAddr

	// SerfLANConfig is the configuration for the intra-dc serf
	SerfLANConfig *serf.Config

	// SerfWANConfig is the configuration for the cross-dc serf
	SerfWANConfig *serf.Config

	// ReconcileInterval controls how often we reconcile the strongly
	// consistent store with the Serf info. This is used to handle nodes
	// that are force removed, as well as intermittent unavailability during
	// leader election.
	ReconcileInterval time.Duration

	// LogOutput is the location to write logs to. If this is not set,
	// logs will go to stderr.
	LogOutput io.Writer

	// ProtocolVersion is the protocol version to speak. This must be between
	// ProtocolVersionMin and ProtocolVersionMax.
	ProtocolVersion uint8

	// VerifyIncoming is used to verify the authenticity of incoming connections.
	// This means that TCP requests are forbidden, only allowing for TLS. TLS connections
	// must match a provided certificate authority. This can be used to force client auth.
	VerifyIncoming bool

	// VerifyOutgoing is used to verify the authenticity of outgoing connections.
	// This means that TLS requests are used, and TCP requests are not made. TLS connections
	// must match a provided certificate authority. This is used to verify authenticity of
	// server nodes.
	VerifyOutgoing bool

	// VerifyServerHostname is used to enable hostname verification of servers. This
	// ensures that the certificate presented is valid for server.<datacenter>.<domain>.
	// This prevents a compromised client from being restarted as a server, and then
	// intercepting request traffic as well as being added as a raft peer. This should be
	// enabled by default with VerifyOutgoing, but for legacy reasons we cannot break
	// existing clients.
	VerifyServerHostname bool

	// CAFile is a path to a certificate authority file. This is used with VerifyIncoming
	// or VerifyOutgoing to verify the TLS connection.
	CAFile string

	// CertFile is used to provide a TLS certificate that is used for serving TLS connections.
	// Must be provided to serve TLS connections.
	CertFile string

	// KeyFile is used to provide a TLS key that is used for serving TLS connections.
	// Must be provided to serve TLS connections.
	KeyFile string

	// ServerName is used with the TLS certificate to ensure the name we
	// provide matches the certificate
	ServerName string

	// RejoinAfterLeave controls our interaction with Serf.
	// When set to false (default), a leave causes a Consul to not rejoin
	// the cluster until an explicit join is received. If this is set to
	// true, we ignore the leave, and rejoin the cluster on start.
	RejoinAfterLeave bool

	// Build is a string that is gossiped around, and can be used to help
	// operators track which versions are actively deployed
	Build string

	// ACLToken is the default token to use when making a request.
	// If not provided, the anonymous token is used. This enables
	// backwards compatibility as well.
	ACLToken string

	// ACLMasterToken is used to bootstrap the ACL system. It should be specified
	// on the servers in the ACLDatacenter. When the leader comes online, it ensures
	// that the Master token is available. This provides the initial token.
	ACLMasterToken string

	// ACLDatacenter provides the authoritative datacenter for ACL
	// tokens. If not provided, ACL verification is disabled.
	ACLDatacenter string

	// ACLTTL controls the time-to-live of cached ACL policies.
	// It can be set to zero to disable caching, but this adds
	// a substantial cost.
	ACLTTL time.Duration

	// ACLDefaultPolicy is used to control the ACL interaction when
	// there is no defined policy. This can be "allow" which means
	// ACLs are used to black-list, or "deny" which means ACLs are
	// white-lists.
	ACLDefaultPolicy string

	// ACLDownPolicy controls the behavior of ACLs if the ACLDatacenter
	// cannot be contacted. It can be either "deny" to deny all requests,
	// or "extend-cache" which ignores the ACLCacheInterval and uses
	// cached policies. If a policy is not in the cache, it acts like deny.
	// "allow" can be used to allow all requests. This is not recommended.
	ACLDownPolicy string

	// TombstoneTTL is used to control how long KV tombstones are retained.
	// This provides a window of time where the X-Consul-Index is monotonic.
	// Outside this window, the index may not be monotonic. This is a result
	// of a few trade offs:
	// 1) The index is defined by the data view and not globally. This is a
	// performance optimization that prevents any write from incrementing the
	// index for all data views.
	// 2) Tombstones are not kept indefinitely, since otherwise storage required
	// is also monotonic. This prevents deletes from reducing the disk space
	// used.
	// In theory, neither of these are intrinsic limitations, however for the
	// purposes of building a practical system, they are reasonable trade offs.
	//
	// It is also possible to set this to an incredibly long time, thereby
	// simulating infinite retention. This is not recommended however.
	//
	TombstoneTTL time.Duration

	// TombstoneTTLGranularity is used to control how granular the timers are
	// for the Tombstone GC. This is used to batch the GC of many keys together
	// to reduce overhead. It is unlikely a user would ever need to tune this.
	TombstoneTTLGranularity time.Duration

	// Minimum Session TTL
	SessionTTLMin time.Duration

	// ServerUp callback can be used to trigger a notification that
	// a Consul server is now up and known about.
	ServerUp func()

	// UserEventHandler callback can be used to handle incoming
	// user events. This function should not block.
	UserEventHandler func(serf.UserEvent)

	// DisableCoordinates controls features related to network coordinates.
	DisableCoordinates bool

	// CoordinateUpdatePeriod controls how long a server batches coordinate
	// updates before applying them in a Raft transaction. A larger period
	// leads to fewer Raft transactions, but also the stored coordinates
	// being more stale.
	CoordinateUpdatePeriod time.Duration

	// CoordinateUpdateBatchSize controls the maximum number of updates a
	// server batches before applying them in a Raft transaction.
	CoordinateUpdateBatchSize int

	// CoordinateUpdateMaxBatches controls the maximum number of batches we
	// are willing to apply in one period. After this limit we will issue a
	// warning and discard the remaining updates.
	CoordinateUpdateMaxBatches int
}

// CheckVersion is used to check if the ProtocolVersion is valid
func (c *Config) CheckVersion() error {
	if c.ProtocolVersion < ProtocolVersionMin {
		return fmt.Errorf("Protocol version '%d' too low. Must be in range: [%d, %d]",
			c.ProtocolVersion, ProtocolVersionMin, ProtocolVersionMax)
	} else if c.ProtocolVersion > ProtocolVersionMax {
		return fmt.Errorf("Protocol version '%d' too high. Must be in range: [%d, %d]",
			c.ProtocolVersion, ProtocolVersionMin, ProtocolVersionMax)
	}
	return nil
}

// CheckACL is used to sanity check the ACL configuration
func (c *Config) CheckACL() error {
	switch c.ACLDefaultPolicy {
	case "allow":
	case "deny":
	default:
		return fmt.Errorf("Unsupported default ACL policy: %s", c.ACLDefaultPolicy)
	}
	switch c.ACLDownPolicy {
	case "allow":
	case "deny":
	case "extend-cache":
	default:
		return fmt.Errorf("Unsupported down ACL policy: %s", c.ACLDownPolicy)
	}
	return nil
}

// DefaultConfig is used to return a sane default configuration
func DefaultConfig() *Config {
	hostname, err := os.Hostname()
	if err != nil {
		panic(err)
	}

	conf := &Config{
		Datacenter:              DefaultDC,
		NodeName:                hostname,
		RPCAddr:                 DefaultRPCAddr,
		RaftConfig:              raft.DefaultConfig(),
		SerfLANConfig:           serf.DefaultConfig(),
		SerfWANConfig:           serf.DefaultConfig(),
		ReconcileInterval:       60 * time.Second,
		ProtocolVersion:         ProtocolVersion2Compatible,
		ACLTTL:                  30 * time.Second,
		ACLDefaultPolicy:        "allow",
		ACLDownPolicy:           "extend-cache",
		TombstoneTTL:            15 * time.Minute,
		TombstoneTTLGranularity: 30 * time.Second,
		SessionTTLMin:           10 * time.Second,
		DisableCoordinates:      false,

		// These are tuned to provide a total throughput of 128 updates
		// per second. If you update these, you should update the client-
		// side SyncCoordinateRateTarget parameter accordingly.
		CoordinateUpdatePeriod:     5 * time.Second,
		CoordinateUpdateBatchSize:  128,
		CoordinateUpdateMaxBatches: 5,
	}

	// Increase our reap interval to 3 days instead of 24h.
	conf.SerfLANConfig.ReconnectTimeout = 3 * 24 * time.Hour
	conf.SerfWANConfig.ReconnectTimeout = 3 * 24 * time.Hour

	// WAN Serf should use the WAN timing, since we are using it
	// to communicate between DC's
	conf.SerfWANConfig.MemberlistConfig = memberlist.DefaultWANConfig()

	// Ensure we don't have port conflicts
	conf.SerfLANConfig.MemberlistConfig.BindPort = DefaultLANSerfPort
	conf.SerfWANConfig.MemberlistConfig.BindPort = DefaultWANSerfPort

	// Disable shutdown on removal
	conf.RaftConfig.ShutdownOnRemove = false

	return conf
}

func (c *Config) tlsConfig() *tlsutil.Config {
	tlsConf := &tlsutil.Config{
		VerifyIncoming:       c.VerifyIncoming,
		VerifyOutgoing:       c.VerifyOutgoing,
		VerifyServerHostname: c.VerifyServerHostname,
		CAFile:               c.CAFile,
		CertFile:             c.CertFile,
		KeyFile:              c.KeyFile,
		NodeName:             c.NodeName,
		ServerName:           c.ServerName,
		Domain:               c.Domain,
	}
	return tlsConf
}
