package storageos

import (
	"encoding/json"
	"net/http"
	"path"

	"github.com/storageos/go-api/types"
)

var (
	// NetworkDiagnosticsAPIPrefix is a partial path to the HTTP endpoint for
	// the node connectivity diagnostics report.
	NetworkDiagnosticsAPIPrefix = "diagnostics/network"
)

// NetworkDiagnostics returns a collection of network connectivity reports.  If
// a reference to a node is given, it will only check connectivity from that
// node.  Otherwise, connectivity between all cluster nodes will be returned.
func (c *Client) NetworkDiagnostics(ref string) (types.ConnectivityResults, error) {
	resp, err := c.do("GET", path.Join(NetworkDiagnosticsAPIPrefix, ref), doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchNode
		}
		return nil, err
	}
	defer resp.Body.Close()

	var results types.ConnectivityResults
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, err
	}
	return results, nil
}
