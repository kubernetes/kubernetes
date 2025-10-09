/*
 *
 * Copyright 2021 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package google

import (
	"context"
	"net"
	"net/url"
	"strings"

	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/internal/xds"
)

const cfeClusterNamePrefix = "google_cfe_"
const cfeClusterResourceNamePrefix = "/envoy.config.cluster.v3.Cluster/google_cfe_"
const cfeClusterAuthorityName = "traffic-director-c2p.xds.googleapis.com"

// clusterTransportCreds is a combo of TLS + ALTS.
//
// On the client, ClientHandshake picks TLS or ALTS based on address attributes.
// - if attributes has cluster name
//   - if cluster name has prefix "google_cfe_", or
//     "xdstp://traffic-director-c2p.xds.googleapis.com/envoy.config.cluster.v3.Cluster/google_cfe_",
//     use TLS
//   - otherwise, use ALTS
//
// - else, do TLS
//
// On the server, ServerHandshake always does TLS.
type clusterTransportCreds struct {
	tls  credentials.TransportCredentials
	alts credentials.TransportCredentials
}

func newClusterTransportCreds(tls, alts credentials.TransportCredentials) *clusterTransportCreds {
	return &clusterTransportCreds{
		tls:  tls,
		alts: alts,
	}
}

// clusterName returns the xDS cluster name stored in the attributes in the
// context.
func clusterName(ctx context.Context) string {
	chi := credentials.ClientHandshakeInfoFromContext(ctx)
	if chi.Attributes == nil {
		return ""
	}
	cluster, _ := xds.GetXDSHandshakeClusterName(chi.Attributes)
	return cluster
}

// isDirectPathCluster returns true if the cluster in the context is a
// directpath cluster, meaning ALTS should be used.
func isDirectPathCluster(ctx context.Context) bool {
	cluster := clusterName(ctx)
	if cluster == "" {
		// No cluster; not xDS; use TLS.
		return false
	}
	if strings.HasPrefix(cluster, cfeClusterNamePrefix) {
		// xDS cluster prefixed by "google_cfe_"; use TLS.
		return false
	}
	if !strings.HasPrefix(cluster, "xdstp:") {
		// Other xDS cluster name; use ALTS.
		return true
	}
	u, err := url.Parse(cluster)
	if err != nil {
		// Shouldn't happen, but assume ALTS.
		return true
	}
	// If authority AND path match our CFE checks, use TLS; otherwise use ALTS.
	return u.Host != cfeClusterAuthorityName || !strings.HasPrefix(u.Path, cfeClusterResourceNamePrefix)
}

func (c *clusterTransportCreds) ClientHandshake(ctx context.Context, authority string, rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	if isDirectPathCluster(ctx) {
		// If attributes have cluster name, and cluster name is not cfe, it's a
		// backend address, use ALTS.
		return c.alts.ClientHandshake(ctx, authority, rawConn)
	}
	return c.tls.ClientHandshake(ctx, authority, rawConn)
}

func (c *clusterTransportCreds) ServerHandshake(conn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return c.tls.ServerHandshake(conn)
}

func (c *clusterTransportCreds) Info() credentials.ProtocolInfo {
	// TODO: this always returns tls.Info now, because we don't have a cluster
	// name to check when this method is called. This method doesn't affect
	// anything important now. We may want to revisit this if it becomes more
	// important later.
	return c.tls.Info()
}

func (c *clusterTransportCreds) Clone() credentials.TransportCredentials {
	return &clusterTransportCreds{
		tls:  c.tls.Clone(),
		alts: c.alts.Clone(),
	}
}

func (c *clusterTransportCreds) OverrideServerName(s string) error {
	if err := c.tls.OverrideServerName(s); err != nil {
		return err
	}
	return c.alts.OverrideServerName(s)
}
