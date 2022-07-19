package clusters

import (
	"strings"

	"github.com/kcp-dev/logicalcluster/v2"
)

// ToClusterAwareKey allows combining the object name and
// the object cluster in a single key that can be used by informers.
// This is KCP-related hack useful when watching across several
// logical clusters using a wildcard context cluster
//
// This is a temporary hack and should be replaced by thoughtful
// and real support of logical cluster in the client-go layer
func ToClusterAwareKey(clusterName logicalcluster.Name, name string) string {
	if !clusterName.Empty() {
		return clusterName.String() + "|" + name
	}

	return name
}

// SplitClusterAwareKey just allows extract the name and clusterName
// from a Key initially created with ToClusterAwareKey
func SplitClusterAwareKey(clusterAwareKey string) (clusterName logicalcluster.Name, name string) {
	parts := strings.SplitN(clusterAwareKey, "|", 2)
	if len(parts) == 1 {
		// name only, no cluster
		return logicalcluster.Name{}, parts[0]
	}
	// clusterName and name
	return logicalcluster.New(parts[0]), parts[1]
}
