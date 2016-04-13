/*
Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package bigtable

import (
	"fmt"
	"regexp"
	"strings"

	"golang.org/x/net/context"
	"google.golang.org/cloud"
	btcspb "google.golang.org/cloud/bigtable/internal/cluster_service_proto"
	bttspb "google.golang.org/cloud/bigtable/internal/table_service_proto"
	"google.golang.org/cloud/internal/transport"
	"google.golang.org/grpc"
)

const adminAddr = "bigtabletableadmin.googleapis.com:443"

// AdminClient is a client type for performing admin operations within a specific cluster.
type AdminClient struct {
	conn    *grpc.ClientConn
	tClient bttspb.BigtableTableServiceClient

	project, zone, cluster string
}

// NewAdminClient creates a new AdminClient for a given project, zone and cluster.
func NewAdminClient(ctx context.Context, project, zone, cluster string, opts ...cloud.ClientOption) (*AdminClient, error) {
	o := []cloud.ClientOption{
		cloud.WithEndpoint(adminAddr),
		cloud.WithScopes(AdminScope),
		cloud.WithUserAgent(clientUserAgent),
	}
	o = append(o, opts...)
	conn, err := transport.DialGRPC(ctx, o...)
	if err != nil {
		return nil, fmt.Errorf("dialing: %v", err)
	}
	return &AdminClient{
		conn:    conn,
		tClient: bttspb.NewBigtableTableServiceClient(conn),

		project: project,
		zone:    zone,
		cluster: cluster,
	}, nil
}

// Close closes the AdminClient.
func (ac *AdminClient) Close() {
	ac.conn.Close()
}

func (ac *AdminClient) clusterPrefix() string {
	return fmt.Sprintf("projects/%s/zones/%s/clusters/%s", ac.project, ac.zone, ac.cluster)
}

// Tables returns a list of the tables in the cluster.
func (ac *AdminClient) Tables(ctx context.Context) ([]string, error) {
	prefix := ac.clusterPrefix()
	req := &bttspb.ListTablesRequest{
		Name: prefix,
	}
	res, err := ac.tClient.ListTables(ctx, req)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(res.Tables))
	for _, tbl := range res.Tables {
		names = append(names, strings.TrimPrefix(tbl.Name, prefix+"/tables/"))
	}
	return names, nil
}

// CreateTable creates a new table in the cluster.
// This method may return before the table's creation is complete.
func (ac *AdminClient) CreateTable(ctx context.Context, table string) error {
	prefix := ac.clusterPrefix()
	req := &bttspb.CreateTableRequest{
		Name:    prefix,
		TableId: table,
	}
	_, err := ac.tClient.CreateTable(ctx, req)
	if err != nil {
		return err
	}
	return nil
}

// CreateColumnFamily creates a new column family in a table.
func (ac *AdminClient) CreateColumnFamily(ctx context.Context, table, family string) error {
	// TODO(dsymonds): Permit specifying gcexpr and any other family settings.
	prefix := ac.clusterPrefix()
	req := &bttspb.CreateColumnFamilyRequest{
		Name:           prefix + "/tables/" + table,
		ColumnFamilyId: family,
	}
	_, err := ac.tClient.CreateColumnFamily(ctx, req)
	return err
}

// DeleteTable deletes a table and all of its data.
func (ac *AdminClient) DeleteTable(ctx context.Context, table string) error {
	prefix := ac.clusterPrefix()
	req := &bttspb.DeleteTableRequest{
		Name: prefix + "/tables/" + table,
	}
	_, err := ac.tClient.DeleteTable(ctx, req)
	return err
}

// DeleteColumnFamily deletes a column family in a table and all of its data.
func (ac *AdminClient) DeleteColumnFamily(ctx context.Context, table, family string) error {
	prefix := ac.clusterPrefix()
	req := &bttspb.DeleteColumnFamilyRequest{
		Name: prefix + "/tables/" + table + "/columnFamilies/" + family,
	}
	_, err := ac.tClient.DeleteColumnFamily(ctx, req)
	return err
}

// TableInfo represents information about a table.
type TableInfo struct {
	Families []string
}

// TableInfo retrieves information about a table.
func (ac *AdminClient) TableInfo(ctx context.Context, table string) (*TableInfo, error) {
	prefix := ac.clusterPrefix()
	req := &bttspb.GetTableRequest{
		Name: prefix + "/tables/" + table,
	}
	res, err := ac.tClient.GetTable(ctx, req)
	if err != nil {
		return nil, err
	}
	ti := &TableInfo{}
	for fam := range res.ColumnFamilies {
		ti.Families = append(ti.Families, fam)
	}
	return ti, nil
}

// SetGCPolicy specifies which cells in a column family should be garbage collected.
// GC executes opportunistically in the background; table reads may return data
// matching the GC policy.
func (ac *AdminClient) SetGCPolicy(ctx context.Context, table, family string, policy GCPolicy) error {
	prefix := ac.clusterPrefix()
	tbl, err := ac.tClient.GetTable(ctx, &bttspb.GetTableRequest{
		Name: prefix + "/tables/" + table,
	})
	if err != nil {
		return err
	}
	fam, ok := tbl.ColumnFamilies[family]
	if !ok {
		return fmt.Errorf("unknown column family %q", family)
	}
	fam.GcRule = policy.proto()
	_, err = ac.tClient.UpdateColumnFamily(ctx, fam)
	return err
}

const clusterAdminAddr = "bigtableclusteradmin.googleapis.com:443"

// ClusterAdminClient is a client type for performing admin operations on clusters.
// These operations can be substantially more dangerous than those provided by AdminClient.
type ClusterAdminClient struct {
	conn    *grpc.ClientConn
	cClient btcspb.BigtableClusterServiceClient

	project string
}

// NewClusterAdminClient creates a new ClusterAdminClient for a given project.
func NewClusterAdminClient(ctx context.Context, project string, opts ...cloud.ClientOption) (*ClusterAdminClient, error) {
	o := []cloud.ClientOption{
		cloud.WithEndpoint(clusterAdminAddr),
		cloud.WithScopes(ClusterAdminScope),
		cloud.WithUserAgent(clientUserAgent),
	}
	o = append(o, opts...)
	conn, err := transport.DialGRPC(ctx, o...)
	if err != nil {
		return nil, fmt.Errorf("dialing: %v", err)
	}
	return &ClusterAdminClient{
		conn:    conn,
		cClient: btcspb.NewBigtableClusterServiceClient(conn),

		project: project,
	}, nil
}

// Close closes the ClusterAdminClient.
func (cac *ClusterAdminClient) Close() {
	cac.conn.Close()
}

// ClusterInfo represents information about a cluster.
type ClusterInfo struct {
	Name        string // name of the cluster
	Zone        string // GCP zone of the cluster (e.g. "us-central1-a")
	DisplayName string // display name for UIs
	ServeNodes  int    // number of allocated serve nodes
}

var clusterNameRegexp = regexp.MustCompile(`^projects/([^/]+)/zones/([^/]+)/clusters/([a-z][-a-z0-9]*)$`)

// Clusters returns a list of clusters in the project.
func (cac *ClusterAdminClient) Clusters(ctx context.Context) ([]*ClusterInfo, error) {
	req := &btcspb.ListClustersRequest{
		Name: "projects/" + cac.project,
	}
	res, err := cac.cClient.ListClusters(ctx, req)
	if err != nil {
		return nil, err
	}
	// TODO(dsymonds): Deal with failed_zones.
	var cis []*ClusterInfo
	for _, c := range res.Clusters {
		m := clusterNameRegexp.FindStringSubmatch(c.Name)
		if m == nil {
			return nil, fmt.Errorf("malformed cluster name %q", c.Name)
		}
		cis = append(cis, &ClusterInfo{
			Name:        m[3],
			Zone:        m[2],
			DisplayName: c.DisplayName,
			ServeNodes:  int(c.ServeNodes),
		})
	}
	return cis, nil
}

/* TODO(dsymonds): Re-enable when there's a ClusterAdmin API.

// SetClusterSize sets the number of server nodes for this cluster.
func (ac *AdminClient) SetClusterSize(ctx context.Context, nodes int) error {
	req := &btcspb.GetClusterRequest{
		Name: ac.clusterPrefix(),
	}
	clu, err := ac.cClient.GetCluster(ctx, req)
	if err != nil {
		return err
	}
	clu.ServeNodes = int32(nodes)
	_, err = ac.cClient.UpdateCluster(ctx, clu)
	return err
}

*/
