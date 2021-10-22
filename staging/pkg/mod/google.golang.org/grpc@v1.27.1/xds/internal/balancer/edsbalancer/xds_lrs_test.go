/*
 *
 * Copyright 2019 gRPC authors.
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

package edsbalancer

import (
	"testing"

	"google.golang.org/grpc/attributes"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/resolver"
	xdsinternal "google.golang.org/grpc/xds/internal"
	"google.golang.org/grpc/xds/internal/testutils/fakeclient"
)

// TestXDSLoadReporting verifies that the edsBalancer starts the loadReport
// stream when the lbConfig passed to it contains a valid value for the LRS
// server (empty string).
func (s) TestXDSLoadReporting(t *testing.T) {
	builder := balancer.Get(edsName)
	cc := newNoopTestClientConn()
	edsB, ok := builder.Build(cc, balancer.BuildOptions{Target: resolver.Target{Endpoint: testEDSClusterName}}).(*edsBalancer)
	if !ok {
		t.Fatalf("builder.Build(%s) returned type {%T}, want {*edsBalancer}", edsName, edsB)
	}
	defer edsB.Close()

	xdsC := fakeclient.NewClient()
	edsB.UpdateClientConnState(balancer.ClientConnState{
		ResolverState:  resolver.State{Attributes: attributes.New(xdsinternal.XDSClientID, xdsC)},
		BalancerConfig: &EDSConfig{LrsLoadReportingServerName: new(string)},
	})

	gotCluster, err := xdsC.WaitForWatchEDS()
	if err != nil {
		t.Fatalf("xdsClient.WatchEDS failed with error: %v", err)
	}
	if gotCluster != testEDSClusterName {
		t.Fatalf("xdsClient.WatchEDS() called with cluster: %v, want %v", gotCluster, testEDSClusterName)
	}

	got, err := xdsC.WaitForReportLoad()
	if err != nil {
		t.Fatalf("xdsClient.ReportLoad failed with error: %v", err)
	}
	if got.Server != "" || got.Cluster != testEDSClusterName {
		t.Fatalf("xdsClient.ReportLoad called with {%v, %v}: want {\"\", %v}", got.Server, got.Cluster, testEDSClusterName)
	}
}
