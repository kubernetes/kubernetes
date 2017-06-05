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

	btopt "cloud.google.com/go/bigtable/internal/option"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
	"google.golang.org/api/transport"
	btapb "google.golang.org/genproto/googleapis/bigtable/admin/v2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

const adminAddr = "bigtableadmin.googleapis.com:443"

// AdminClient is a client type for performing admin operations within a specific instance.
type AdminClient struct {
	conn    *grpc.ClientConn
	tClient btapb.BigtableTableAdminClient

	project, instance string

	// Metadata to be sent with each request.
	md metadata.MD
}

// NewAdminClient creates a new AdminClient for a given project and instance.
func NewAdminClient(ctx context.Context, project, instance string, opts ...option.ClientOption) (*AdminClient, error) {
	o, err := btopt.DefaultClientOptions(adminAddr, AdminScope, clientUserAgent)
	if err != nil {
		return nil, err
	}
	o = append(o, opts...)
	conn, err := transport.DialGRPC(ctx, o...)
	if err != nil {
		return nil, fmt.Errorf("dialing: %v", err)
	}
	return &AdminClient{
		conn:     conn,
		tClient:  btapb.NewBigtableTableAdminClient(conn),
		project:  project,
		instance: instance,
		md:       metadata.Pairs(resourcePrefixHeader, fmt.Sprintf("projects/%s/instances/%s", project, instance)),
	}, nil
}

// Close closes the AdminClient.
func (ac *AdminClient) Close() error {
	return ac.conn.Close()
}

func (ac *AdminClient) instancePrefix() string {
	return fmt.Sprintf("projects/%s/instances/%s", ac.project, ac.instance)
}

// Tables returns a list of the tables in the instance.
func (ac *AdminClient) Tables(ctx context.Context) ([]string, error) {
	ctx = metadata.NewContext(ctx, ac.md)
	prefix := ac.instancePrefix()
	req := &btapb.ListTablesRequest{
		Parent: prefix,
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

// CreateTable creates a new table in the instance.
// This method may return before the table's creation is complete.
func (ac *AdminClient) CreateTable(ctx context.Context, table string) error {
	ctx = metadata.NewContext(ctx, ac.md)
	prefix := ac.instancePrefix()
	req := &btapb.CreateTableRequest{
		Parent:  prefix,
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
	ctx = metadata.NewContext(ctx, ac.md)
	prefix := ac.instancePrefix()
	req := &btapb.ModifyColumnFamiliesRequest{
		Name: prefix + "/tables/" + table,
		Modifications: []*btapb.ModifyColumnFamiliesRequest_Modification{
			{
				Id:  family,
				Mod: &btapb.ModifyColumnFamiliesRequest_Modification_Create{Create: &btapb.ColumnFamily{}},
			},
		},
	}
	_, err := ac.tClient.ModifyColumnFamilies(ctx, req)
	return err
}

// DeleteTable deletes a table and all of its data.
func (ac *AdminClient) DeleteTable(ctx context.Context, table string) error {
	ctx = metadata.NewContext(ctx, ac.md)
	prefix := ac.instancePrefix()
	req := &btapb.DeleteTableRequest{
		Name: prefix + "/tables/" + table,
	}
	_, err := ac.tClient.DeleteTable(ctx, req)
	return err
}

// DeleteColumnFamily deletes a column family in a table and all of its data.
func (ac *AdminClient) DeleteColumnFamily(ctx context.Context, table, family string) error {
	ctx = metadata.NewContext(ctx, ac.md)
	prefix := ac.instancePrefix()
	req := &btapb.ModifyColumnFamiliesRequest{
		Name: prefix + "/tables/" + table,
		Modifications: []*btapb.ModifyColumnFamiliesRequest_Modification{
			{
				Id:  family,
				Mod: &btapb.ModifyColumnFamiliesRequest_Modification_Drop{Drop: true},
			},
		},
	}
	_, err := ac.tClient.ModifyColumnFamilies(ctx, req)
	return err
}

// TableInfo represents information about a table.
type TableInfo struct {
	Families []string
}

// TableInfo retrieves information about a table.
func (ac *AdminClient) TableInfo(ctx context.Context, table string) (*TableInfo, error) {
	ctx = metadata.NewContext(ctx, ac.md)
	prefix := ac.instancePrefix()
	req := &btapb.GetTableRequest{
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
	ctx = metadata.NewContext(ctx, ac.md)
	prefix := ac.instancePrefix()
	req := &btapb.ModifyColumnFamiliesRequest{
		Name: prefix + "/tables/" + table,
		Modifications: []*btapb.ModifyColumnFamiliesRequest_Modification{
			{
				Id:  family,
				Mod: &btapb.ModifyColumnFamiliesRequest_Modification_Update{Update: &btapb.ColumnFamily{GcRule: policy.proto()}},
			},
		},
	}
	_, err := ac.tClient.ModifyColumnFamilies(ctx, req)
	return err
}

const instanceAdminAddr = "bigtableadmin.googleapis.com:443"

// InstanceAdminClient is a client type for performing admin operations on instances.
// These operations can be substantially more dangerous than those provided by AdminClient.
type InstanceAdminClient struct {
	conn    *grpc.ClientConn
	iClient btapb.BigtableInstanceAdminClient

	project string

	// Metadata to be sent with each request.
	md metadata.MD
}

// NewInstanceAdminClient creates a new InstanceAdminClient for a given project.
func NewInstanceAdminClient(ctx context.Context, project string, opts ...option.ClientOption) (*InstanceAdminClient, error) {
	o, err := btopt.DefaultClientOptions(instanceAdminAddr, InstanceAdminScope, clientUserAgent)
	if err != nil {
		return nil, err
	}
	o = append(o, opts...)
	conn, err := transport.DialGRPC(ctx, o...)
	if err != nil {
		return nil, fmt.Errorf("dialing: %v", err)
	}
	return &InstanceAdminClient{
		conn:    conn,
		iClient: btapb.NewBigtableInstanceAdminClient(conn),

		project: project,
		md:      metadata.Pairs(resourcePrefixHeader, "projects/"+project),
	}, nil
}

// Close closes the InstanceAdminClient.
func (iac *InstanceAdminClient) Close() error {
	return iac.conn.Close()
}

// InstanceInfo represents information about an instance
type InstanceInfo struct {
	Name        string // name of the instance
	DisplayName string // display name for UIs
}

var instanceNameRegexp = regexp.MustCompile(`^projects/([^/]+)/instances/([a-z][-a-z0-9]*)$`)

// Instances returns a list of instances in the project.
func (cac *InstanceAdminClient) Instances(ctx context.Context) ([]*InstanceInfo, error) {
	ctx = metadata.NewContext(ctx, cac.md)
	req := &btapb.ListInstancesRequest{
		Parent: "projects/" + cac.project,
	}
	res, err := cac.iClient.ListInstances(ctx, req)
	if err != nil {
		return nil, err
	}

	var is []*InstanceInfo
	for _, i := range res.Instances {
		m := instanceNameRegexp.FindStringSubmatch(i.Name)
		if m == nil {
			return nil, fmt.Errorf("malformed instance name %q", i.Name)
		}
		is = append(is, &InstanceInfo{
			Name:        m[2],
			DisplayName: i.DisplayName,
		})
	}
	return is, nil
}
