/*
 *
 * Copyright 2020 gRPC authors.
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

package clustermanager

import (
	"context"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// pickerGroup contains a list of pickers. If the picker isn't ready, the pick
// will be queued.
type pickerGroup struct {
	pickers map[string]balancer.Picker
}

func newPickerGroup(idToPickerState map[string]*subBalancerState) *pickerGroup {
	pickers := make(map[string]balancer.Picker)
	for id, st := range idToPickerState {
		pickers[id] = st.state.Picker
	}
	return &pickerGroup{
		pickers: pickers,
	}
}

func (pg *pickerGroup) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	cluster := getPickedCluster(info.Ctx)
	if p := pg.pickers[cluster]; p != nil {
		return p.Pick(info)
	}
	return balancer.PickResult{}, status.Errorf(codes.Unavailable, "unknown cluster selected for RPC: %q", cluster)
}

type clusterKey struct{}

func getPickedCluster(ctx context.Context) string {
	cluster, _ := ctx.Value(clusterKey{}).(string)
	return cluster
}

// GetPickedClusterForTesting returns the cluster in the context; to be used
// for testing only.
func GetPickedClusterForTesting(ctx context.Context) string {
	return getPickedCluster(ctx)
}

// SetPickedCluster adds the selected cluster to the context for the
// xds_cluster_manager LB policy to pick.
func SetPickedCluster(ctx context.Context, cluster string) context.Context {
	return context.WithValue(ctx, clusterKey{}, cluster)
}
