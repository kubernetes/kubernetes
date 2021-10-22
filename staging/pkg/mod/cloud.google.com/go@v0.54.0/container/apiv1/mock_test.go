// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// AUTO-GENERATED CODE. DO NOT EDIT.

package container

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes"
	emptypb "github.com/golang/protobuf/ptypes/empty"
	"google.golang.org/api/option"
	containerpb "google.golang.org/genproto/googleapis/container/v1"

	status "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"

	gstatus "google.golang.org/grpc/status"
)

var _ = io.EOF
var _ = ptypes.MarshalAny
var _ status.Status

type mockClusterManagerServer struct {
	// Embed for forward compatibility.
	// Tests will keep working if more methods are added
	// in the future.
	containerpb.ClusterManagerServer

	reqs []proto.Message

	// If set, all calls return this error.
	err error

	// responses to return if err == nil
	resps []proto.Message
}

func (s *mockClusterManagerServer) ListClusters(ctx context.Context, req *containerpb.ListClustersRequest) (*containerpb.ListClustersResponse, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.ListClustersResponse), nil
}

func (s *mockClusterManagerServer) GetCluster(ctx context.Context, req *containerpb.GetClusterRequest) (*containerpb.Cluster, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Cluster), nil
}

func (s *mockClusterManagerServer) CreateCluster(ctx context.Context, req *containerpb.CreateClusterRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) UpdateCluster(ctx context.Context, req *containerpb.UpdateClusterRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) UpdateNodePool(ctx context.Context, req *containerpb.UpdateNodePoolRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetNodePoolAutoscaling(ctx context.Context, req *containerpb.SetNodePoolAutoscalingRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetLoggingService(ctx context.Context, req *containerpb.SetLoggingServiceRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetMonitoringService(ctx context.Context, req *containerpb.SetMonitoringServiceRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetAddonsConfig(ctx context.Context, req *containerpb.SetAddonsConfigRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetLocations(ctx context.Context, req *containerpb.SetLocationsRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) UpdateMaster(ctx context.Context, req *containerpb.UpdateMasterRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetMasterAuth(ctx context.Context, req *containerpb.SetMasterAuthRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) DeleteCluster(ctx context.Context, req *containerpb.DeleteClusterRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) ListOperations(ctx context.Context, req *containerpb.ListOperationsRequest) (*containerpb.ListOperationsResponse, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.ListOperationsResponse), nil
}

func (s *mockClusterManagerServer) GetOperation(ctx context.Context, req *containerpb.GetOperationRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) CancelOperation(ctx context.Context, req *containerpb.CancelOperationRequest) (*emptypb.Empty, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*emptypb.Empty), nil
}

func (s *mockClusterManagerServer) GetServerConfig(ctx context.Context, req *containerpb.GetServerConfigRequest) (*containerpb.ServerConfig, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.ServerConfig), nil
}

func (s *mockClusterManagerServer) ListNodePools(ctx context.Context, req *containerpb.ListNodePoolsRequest) (*containerpb.ListNodePoolsResponse, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.ListNodePoolsResponse), nil
}

func (s *mockClusterManagerServer) GetNodePool(ctx context.Context, req *containerpb.GetNodePoolRequest) (*containerpb.NodePool, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.NodePool), nil
}

func (s *mockClusterManagerServer) CreateNodePool(ctx context.Context, req *containerpb.CreateNodePoolRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) DeleteNodePool(ctx context.Context, req *containerpb.DeleteNodePoolRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) RollbackNodePoolUpgrade(ctx context.Context, req *containerpb.RollbackNodePoolUpgradeRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetNodePoolManagement(ctx context.Context, req *containerpb.SetNodePoolManagementRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetLabels(ctx context.Context, req *containerpb.SetLabelsRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetLegacyAbac(ctx context.Context, req *containerpb.SetLegacyAbacRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) StartIPRotation(ctx context.Context, req *containerpb.StartIPRotationRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) CompleteIPRotation(ctx context.Context, req *containerpb.CompleteIPRotationRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetNodePoolSize(ctx context.Context, req *containerpb.SetNodePoolSizeRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetNetworkPolicy(ctx context.Context, req *containerpb.SetNetworkPolicyRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

func (s *mockClusterManagerServer) SetMaintenancePolicy(ctx context.Context, req *containerpb.SetMaintenancePolicyRequest) (*containerpb.Operation, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	if xg := md["x-goog-api-client"]; len(xg) == 0 || !strings.Contains(xg[0], "gl-go/") {
		return nil, fmt.Errorf("x-goog-api-client = %v, expected gl-go key", xg)
	}
	s.reqs = append(s.reqs, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resps[0].(*containerpb.Operation), nil
}

// clientOpt is the option tests should use to connect to the test server.
// It is initialized by TestMain.
var clientOpt option.ClientOption

var (
	mockClusterManager mockClusterManagerServer
)

func TestMain(m *testing.M) {
	flag.Parse()

	serv := grpc.NewServer()
	containerpb.RegisterClusterManagerServer(serv, &mockClusterManager)

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		log.Fatal(err)
	}
	go serv.Serve(lis)

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithInsecure())
	if err != nil {
		log.Fatal(err)
	}
	clientOpt = option.WithGRPCConn(conn)

	os.Exit(m.Run())
}

func TestClusterManagerListClusters(t *testing.T) {
	var expectedResponse *containerpb.ListClustersResponse = &containerpb.ListClustersResponse{}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var request = &containerpb.ListClustersRequest{
		ProjectId: projectId,
		Zone:      zone,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.ListClusters(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerListClustersError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var request = &containerpb.ListClustersRequest{
		ProjectId: projectId,
		Zone:      zone,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.ListClusters(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerGetCluster(t *testing.T) {
	var name string = "name3373707"
	var description string = "description-1724546052"
	var initialNodeCount int32 = 1682564205
	var loggingService string = "loggingService-1700501035"
	var monitoringService string = "monitoringService1469270462"
	var network string = "network1843485230"
	var clusterIpv4Cidr string = "clusterIpv4Cidr-141875831"
	var subnetwork string = "subnetwork-1302785042"
	var enableKubernetesAlpha bool = false
	var labelFingerprint string = "labelFingerprint714995737"
	var selfLink string = "selfLink-1691268851"
	var zone2 string = "zone2-696322977"
	var endpoint string = "endpoint1741102485"
	var initialClusterVersion string = "initialClusterVersion-276373352"
	var currentMasterVersion string = "currentMasterVersion-920953983"
	var currentNodeVersion string = "currentNodeVersion-407476063"
	var createTime string = "createTime-493574096"
	var statusMessage string = "statusMessage-239442758"
	var nodeIpv4CidrSize int32 = 1181176815
	var servicesIpv4Cidr string = "servicesIpv4Cidr1966438125"
	var currentNodeCount int32 = 178977560
	var expireTime string = "expireTime-96179731"
	var expectedResponse = &containerpb.Cluster{
		Name:                  name,
		Description:           description,
		InitialNodeCount:      initialNodeCount,
		LoggingService:        loggingService,
		MonitoringService:     monitoringService,
		Network:               network,
		ClusterIpv4Cidr:       clusterIpv4Cidr,
		Subnetwork:            subnetwork,
		EnableKubernetesAlpha: enableKubernetesAlpha,
		LabelFingerprint:      labelFingerprint,
		SelfLink:              selfLink,
		Zone:                  zone2,
		Endpoint:              endpoint,
		InitialClusterVersion: initialClusterVersion,
		CurrentMasterVersion:  currentMasterVersion,
		CurrentNodeVersion:    currentNodeVersion,
		CreateTime:            createTime,
		StatusMessage:         statusMessage,
		NodeIpv4CidrSize:      nodeIpv4CidrSize,
		ServicesIpv4Cidr:      servicesIpv4Cidr,
		CurrentNodeCount:      currentNodeCount,
		ExpireTime:            expireTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.GetClusterRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.GetCluster(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerGetClusterError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.GetClusterRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.GetCluster(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerCreateCluster(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var cluster *containerpb.Cluster = &containerpb.Cluster{}
	var request = &containerpb.CreateClusterRequest{
		ProjectId: projectId,
		Zone:      zone,
		Cluster:   cluster,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.CreateCluster(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerCreateClusterError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var cluster *containerpb.Cluster = &containerpb.Cluster{}
	var request = &containerpb.CreateClusterRequest{
		ProjectId: projectId,
		Zone:      zone,
		Cluster:   cluster,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.CreateCluster(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerUpdateCluster(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var update *containerpb.ClusterUpdate = &containerpb.ClusterUpdate{}
	var request = &containerpb.UpdateClusterRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		Update:    update,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.UpdateCluster(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerUpdateClusterError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var update *containerpb.ClusterUpdate = &containerpb.ClusterUpdate{}
	var request = &containerpb.UpdateClusterRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		Update:    update,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.UpdateCluster(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerUpdateNodePool(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var nodeVersion string = "nodeVersion1790136219"
	var imageType string = "imageType-1442758754"
	var request = &containerpb.UpdateNodePoolRequest{
		ProjectId:   projectId,
		Zone:        zone,
		ClusterId:   clusterId,
		NodePoolId:  nodePoolId,
		NodeVersion: nodeVersion,
		ImageType:   imageType,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.UpdateNodePool(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerUpdateNodePoolError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var nodeVersion string = "nodeVersion1790136219"
	var imageType string = "imageType-1442758754"
	var request = &containerpb.UpdateNodePoolRequest{
		ProjectId:   projectId,
		Zone:        zone,
		ClusterId:   clusterId,
		NodePoolId:  nodePoolId,
		NodeVersion: nodeVersion,
		ImageType:   imageType,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.UpdateNodePool(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetNodePoolAutoscaling(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var autoscaling *containerpb.NodePoolAutoscaling = &containerpb.NodePoolAutoscaling{}
	var request = &containerpb.SetNodePoolAutoscalingRequest{
		ProjectId:   projectId,
		Zone:        zone,
		ClusterId:   clusterId,
		NodePoolId:  nodePoolId,
		Autoscaling: autoscaling,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetNodePoolAutoscaling(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetNodePoolAutoscalingError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var autoscaling *containerpb.NodePoolAutoscaling = &containerpb.NodePoolAutoscaling{}
	var request = &containerpb.SetNodePoolAutoscalingRequest{
		ProjectId:   projectId,
		Zone:        zone,
		ClusterId:   clusterId,
		NodePoolId:  nodePoolId,
		Autoscaling: autoscaling,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetNodePoolAutoscaling(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetLoggingService(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var loggingService string = "loggingService-1700501035"
	var request = &containerpb.SetLoggingServiceRequest{
		ProjectId:      projectId,
		Zone:           zone,
		ClusterId:      clusterId,
		LoggingService: loggingService,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetLoggingService(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetLoggingServiceError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var loggingService string = "loggingService-1700501035"
	var request = &containerpb.SetLoggingServiceRequest{
		ProjectId:      projectId,
		Zone:           zone,
		ClusterId:      clusterId,
		LoggingService: loggingService,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetLoggingService(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetMonitoringService(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var monitoringService string = "monitoringService1469270462"
	var request = &containerpb.SetMonitoringServiceRequest{
		ProjectId:         projectId,
		Zone:              zone,
		ClusterId:         clusterId,
		MonitoringService: monitoringService,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetMonitoringService(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetMonitoringServiceError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var monitoringService string = "monitoringService1469270462"
	var request = &containerpb.SetMonitoringServiceRequest{
		ProjectId:         projectId,
		Zone:              zone,
		ClusterId:         clusterId,
		MonitoringService: monitoringService,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetMonitoringService(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetAddonsConfig(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var addonsConfig *containerpb.AddonsConfig = &containerpb.AddonsConfig{}
	var request = &containerpb.SetAddonsConfigRequest{
		ProjectId:    projectId,
		Zone:         zone,
		ClusterId:    clusterId,
		AddonsConfig: addonsConfig,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetAddonsConfig(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetAddonsConfigError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var addonsConfig *containerpb.AddonsConfig = &containerpb.AddonsConfig{}
	var request = &containerpb.SetAddonsConfigRequest{
		ProjectId:    projectId,
		Zone:         zone,
		ClusterId:    clusterId,
		AddonsConfig: addonsConfig,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetAddonsConfig(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetLocations(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var locations []string = nil
	var request = &containerpb.SetLocationsRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		Locations: locations,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetLocations(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetLocationsError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var locations []string = nil
	var request = &containerpb.SetLocationsRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		Locations: locations,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetLocations(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerUpdateMaster(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var masterVersion string = "masterVersion-2139460613"
	var request = &containerpb.UpdateMasterRequest{
		ProjectId:     projectId,
		Zone:          zone,
		ClusterId:     clusterId,
		MasterVersion: masterVersion,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.UpdateMaster(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerUpdateMasterError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var masterVersion string = "masterVersion-2139460613"
	var request = &containerpb.UpdateMasterRequest{
		ProjectId:     projectId,
		Zone:          zone,
		ClusterId:     clusterId,
		MasterVersion: masterVersion,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.UpdateMaster(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetMasterAuth(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var action containerpb.SetMasterAuthRequest_Action = containerpb.SetMasterAuthRequest_UNKNOWN
	var update *containerpb.MasterAuth = &containerpb.MasterAuth{}
	var request = &containerpb.SetMasterAuthRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		Action:    action,
		Update:    update,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetMasterAuth(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetMasterAuthError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var action containerpb.SetMasterAuthRequest_Action = containerpb.SetMasterAuthRequest_UNKNOWN
	var update *containerpb.MasterAuth = &containerpb.MasterAuth{}
	var request = &containerpb.SetMasterAuthRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		Action:    action,
		Update:    update,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetMasterAuth(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerDeleteCluster(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.DeleteClusterRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.DeleteCluster(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerDeleteClusterError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.DeleteClusterRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.DeleteCluster(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerListOperations(t *testing.T) {
	var expectedResponse *containerpb.ListOperationsResponse = &containerpb.ListOperationsResponse{}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var request = &containerpb.ListOperationsRequest{
		ProjectId: projectId,
		Zone:      zone,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.ListOperations(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerListOperationsError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var request = &containerpb.ListOperationsRequest{
		ProjectId: projectId,
		Zone:      zone,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.ListOperations(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerGetOperation(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var operationId string = "operationId-274116877"
	var request = &containerpb.GetOperationRequest{
		ProjectId:   projectId,
		Zone:        zone,
		OperationId: operationId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.GetOperation(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerGetOperationError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var operationId string = "operationId-274116877"
	var request = &containerpb.GetOperationRequest{
		ProjectId:   projectId,
		Zone:        zone,
		OperationId: operationId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.GetOperation(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerCancelOperation(t *testing.T) {
	var expectedResponse *emptypb.Empty = &emptypb.Empty{}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var operationId string = "operationId-274116877"
	var request = &containerpb.CancelOperationRequest{
		ProjectId:   projectId,
		Zone:        zone,
		OperationId: operationId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	err = c.CancelOperation(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

}

func TestClusterManagerCancelOperationError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var operationId string = "operationId-274116877"
	var request = &containerpb.CancelOperationRequest{
		ProjectId:   projectId,
		Zone:        zone,
		OperationId: operationId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	err = c.CancelOperation(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
}
func TestClusterManagerGetServerConfig(t *testing.T) {
	var defaultClusterVersion string = "defaultClusterVersion111003029"
	var defaultImageType string = "defaultImageType-918225828"
	var expectedResponse = &containerpb.ServerConfig{
		DefaultClusterVersion: defaultClusterVersion,
		DefaultImageType:      defaultImageType,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var request = &containerpb.GetServerConfigRequest{
		ProjectId: projectId,
		Zone:      zone,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.GetServerConfig(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerGetServerConfigError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var request = &containerpb.GetServerConfigRequest{
		ProjectId: projectId,
		Zone:      zone,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.GetServerConfig(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerListNodePools(t *testing.T) {
	var expectedResponse *containerpb.ListNodePoolsResponse = &containerpb.ListNodePoolsResponse{}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.ListNodePoolsRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.ListNodePools(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerListNodePoolsError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.ListNodePoolsRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.ListNodePools(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerGetNodePool(t *testing.T) {
	var name string = "name3373707"
	var initialNodeCount int32 = 1682564205
	var selfLink string = "selfLink-1691268851"
	var version string = "version351608024"
	var statusMessage string = "statusMessage-239442758"
	var expectedResponse = &containerpb.NodePool{
		Name:             name,
		InitialNodeCount: initialNodeCount,
		SelfLink:         selfLink,
		Version:          version,
		StatusMessage:    statusMessage,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var request = &containerpb.GetNodePoolRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.GetNodePool(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerGetNodePoolError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var request = &containerpb.GetNodePoolRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.GetNodePool(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerCreateNodePool(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePool *containerpb.NodePool = &containerpb.NodePool{}
	var request = &containerpb.CreateNodePoolRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		NodePool:  nodePool,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.CreateNodePool(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerCreateNodePoolError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePool *containerpb.NodePool = &containerpb.NodePool{}
	var request = &containerpb.CreateNodePoolRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		NodePool:  nodePool,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.CreateNodePool(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerDeleteNodePool(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var request = &containerpb.DeleteNodePoolRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.DeleteNodePool(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerDeleteNodePoolError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var request = &containerpb.DeleteNodePoolRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.DeleteNodePool(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerRollbackNodePoolUpgrade(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var request = &containerpb.RollbackNodePoolUpgradeRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.RollbackNodePoolUpgrade(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerRollbackNodePoolUpgradeError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var request = &containerpb.RollbackNodePoolUpgradeRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.RollbackNodePoolUpgrade(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetNodePoolManagement(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var management *containerpb.NodeManagement = &containerpb.NodeManagement{}
	var request = &containerpb.SetNodePoolManagementRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
		Management: management,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetNodePoolManagement(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetNodePoolManagementError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var management *containerpb.NodeManagement = &containerpb.NodeManagement{}
	var request = &containerpb.SetNodePoolManagementRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
		Management: management,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetNodePoolManagement(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetLabels(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var resourceLabels map[string]string = nil
	var labelFingerprint string = "labelFingerprint714995737"
	var request = &containerpb.SetLabelsRequest{
		ProjectId:        projectId,
		Zone:             zone,
		ClusterId:        clusterId,
		ResourceLabels:   resourceLabels,
		LabelFingerprint: labelFingerprint,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetLabels(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetLabelsError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var resourceLabels map[string]string = nil
	var labelFingerprint string = "labelFingerprint714995737"
	var request = &containerpb.SetLabelsRequest{
		ProjectId:        projectId,
		Zone:             zone,
		ClusterId:        clusterId,
		ResourceLabels:   resourceLabels,
		LabelFingerprint: labelFingerprint,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetLabels(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetLegacyAbac(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var enabled bool = false
	var request = &containerpb.SetLegacyAbacRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		Enabled:   enabled,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetLegacyAbac(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetLegacyAbacError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var enabled bool = false
	var request = &containerpb.SetLegacyAbacRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
		Enabled:   enabled,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetLegacyAbac(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerStartIPRotation(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.StartIPRotationRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.StartIPRotation(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerStartIPRotationError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.StartIPRotationRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.StartIPRotation(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerCompleteIPRotation(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.CompleteIPRotationRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.CompleteIPRotation(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerCompleteIPRotationError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var request = &containerpb.CompleteIPRotationRequest{
		ProjectId: projectId,
		Zone:      zone,
		ClusterId: clusterId,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.CompleteIPRotation(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetNodePoolSize(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var nodeCount int32 = 1539922066
	var request = &containerpb.SetNodePoolSizeRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
		NodeCount:  nodeCount,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetNodePoolSize(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetNodePoolSizeError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var nodePoolId string = "nodePoolId1043384033"
	var nodeCount int32 = 1539922066
	var request = &containerpb.SetNodePoolSizeRequest{
		ProjectId:  projectId,
		Zone:       zone,
		ClusterId:  clusterId,
		NodePoolId: nodePoolId,
		NodeCount:  nodeCount,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetNodePoolSize(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetNetworkPolicy(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var networkPolicy *containerpb.NetworkPolicy = &containerpb.NetworkPolicy{}
	var request = &containerpb.SetNetworkPolicyRequest{
		ProjectId:     projectId,
		Zone:          zone,
		ClusterId:     clusterId,
		NetworkPolicy: networkPolicy,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetNetworkPolicy(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetNetworkPolicyError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var networkPolicy *containerpb.NetworkPolicy = &containerpb.NetworkPolicy{}
	var request = &containerpb.SetNetworkPolicyRequest{
		ProjectId:     projectId,
		Zone:          zone,
		ClusterId:     clusterId,
		NetworkPolicy: networkPolicy,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetNetworkPolicy(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
func TestClusterManagerSetMaintenancePolicy(t *testing.T) {
	var name string = "name3373707"
	var zone2 string = "zone2-696322977"
	var detail string = "detail-1335224239"
	var statusMessage string = "statusMessage-239442758"
	var selfLink string = "selfLink-1691268851"
	var targetLink string = "targetLink-2084812312"
	var startTime string = "startTime-1573145462"
	var endTime string = "endTime1725551537"
	var expectedResponse = &containerpb.Operation{
		Name:          name,
		Zone:          zone2,
		Detail:        detail,
		StatusMessage: statusMessage,
		SelfLink:      selfLink,
		TargetLink:    targetLink,
		StartTime:     startTime,
		EndTime:       endTime,
	}

	mockClusterManager.err = nil
	mockClusterManager.reqs = nil

	mockClusterManager.resps = append(mockClusterManager.resps[:0], expectedResponse)

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var maintenancePolicy *containerpb.MaintenancePolicy = &containerpb.MaintenancePolicy{}
	var request = &containerpb.SetMaintenancePolicyRequest{
		ProjectId:         projectId,
		Zone:              zone,
		ClusterId:         clusterId,
		MaintenancePolicy: maintenancePolicy,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetMaintenancePolicy(context.Background(), request)

	if err != nil {
		t.Fatal(err)
	}

	if want, got := request, mockClusterManager.reqs[0]; !proto.Equal(want, got) {
		t.Errorf("wrong request %q, want %q", got, want)
	}

	if want, got := expectedResponse, resp; !proto.Equal(want, got) {
		t.Errorf("wrong response %q, want %q)", got, want)
	}
}

func TestClusterManagerSetMaintenancePolicyError(t *testing.T) {
	errCode := codes.PermissionDenied
	mockClusterManager.err = gstatus.Error(errCode, "test error")

	var projectId string = "projectId-1969970175"
	var zone string = "zone3744684"
	var clusterId string = "clusterId240280960"
	var maintenancePolicy *containerpb.MaintenancePolicy = &containerpb.MaintenancePolicy{}
	var request = &containerpb.SetMaintenancePolicyRequest{
		ProjectId:         projectId,
		Zone:              zone,
		ClusterId:         clusterId,
		MaintenancePolicy: maintenancePolicy,
	}

	c, err := NewClusterManagerClient(context.Background(), clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := c.SetMaintenancePolicy(context.Background(), request)

	if st, ok := gstatus.FromError(err); !ok {
		t.Errorf("got error %v, expected grpc error", err)
	} else if c := st.Code(); c != errCode {
		t.Errorf("got error code %q, want %q", c, errCode)
	}
	_ = resp
}
