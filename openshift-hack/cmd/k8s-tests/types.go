package main

import (
	"fmt"

	"github.com/onsi/ginkgo/v2/types"
)

// copied directly from github.com/openshift/origin/test/extended/util/cluster/cluster.go
type ClusterConfiguration struct {
	ProviderName string `json:"type"`

	// These fields (and the "type" tag for ProviderName) chosen to match
	// upstream's e2e.CloudConfig.
	ProjectID   string
	Region      string
	Zone        string
	NumNodes    int
	MultiMaster bool
	MultiZone   bool
	Zones       []string
	ConfigFile  string

	// Disconnected is set for test jobs without external internet connectivity
	Disconnected bool

	// SingleReplicaTopology is set for disabling disruptive tests or tests
	// that require high availability
	SingleReplicaTopology bool

	// NetworkPlugin is the "official" plugin name
	NetworkPlugin string
	// NetworkPluginMode is an optional sub-identifier for the NetworkPlugin.
	// (Currently it is only used for OpenShiftSDN.)
	NetworkPluginMode string `json:",omitempty"`

	// HasIPv4 and HasIPv6 determine whether IPv4-specific, IPv6-specific,
	// and dual-stack-specific tests are run
	HasIPv4 bool
	HasIPv6 bool

	// HasSCTP determines whether SCTP connectivity tests can be run in the cluster
	HasSCTP bool

	// IsProxied determines whether we are accessing the cluster through an HTTP proxy
	IsProxied bool

	// IsIBMROKS determines whether the cluster is Managed IBM Cloud (ROKS)
	IsIBMROKS bool

	// IsNoOptionalCapabilities indicates the cluster has no optional capabilities enabled
	HasNoOptionalCapabilities bool
}

// copied directly from github.com/openshift/origin/pkg/test/ginkgo/test.go
type TestCase struct {
	Name      string
	Labels    string
	spec      types.TestSpec
	locations []types.CodeLocation
}

type ExitError struct {
	Code int
}

func (e ExitError) Error() string {
	return fmt.Sprintf("exit with code %d", e.Code)
}
