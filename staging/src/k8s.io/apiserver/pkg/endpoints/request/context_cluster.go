/*
Copyright 2023 The Kubernetes Authors.

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

package request

import (
	"context"
	"errors"
	"fmt"

	"github.com/kcp-dev/logicalcluster/v3"
)

type clusterKey int

const (
	// clusterKey is the context key for the request namespace.
	clusterContextKey clusterKey = iota
)

type Cluster struct {
	// Name holds a cluster name. This is empty for wildcard requests.
	Name logicalcluster.Name

	// If true the query applies to all clusters. Name is empty if this is true.
	Wildcard bool

	// PartialMetadataRequest indicates if the incoming request is for partial metadata. This is set by the kcp
	// server handlers and is necessary to get the right plumbing in place for wildcard partial metadata requests for
	// custom resources.
	PartialMetadataRequest bool
}

// WithCluster returns a context that describes the nested cluster context
func WithCluster(parent context.Context, cluster Cluster) context.Context {
	return context.WithValue(parent, clusterContextKey, cluster)
}

// ClusterFrom returns the value of the cluster key on the ctx, or nil if there
// is no cluster key.
func ClusterFrom(ctx context.Context) *Cluster {
	cluster, ok := ctx.Value(clusterContextKey).(Cluster)
	if !ok {
		return nil
	}
	return &cluster
}

func buildClusterError(message string, ctx context.Context) error {
	if ri, ok := RequestInfoFrom(ctx); ok {
		message = message + fmt.Sprintf(" - RequestInfo: %#v", ri)
	}
	return errors.New(message)
}

// ValidClusterFrom returns the value of the cluster key on the ctx.
// If there's no cluster key, or if the cluster name is empty
// and it's not a wildcard context, then return an error.
func ValidClusterFrom(ctx context.Context) (*Cluster, error) {
	cluster := ClusterFrom(ctx)
	if cluster == nil {
		return nil, buildClusterError("no cluster in the request context", ctx)
	}
	if cluster.Name.Empty() && !cluster.Wildcard {
		return nil, buildClusterError("cluster path is empty in the request context", ctx)
	}
	return cluster, nil
}

// ClusterNameOrWildcardFrom returns a cluster.Name or true for a wildcard from
// the value of the cluster key on the ctx.
func ClusterNameOrWildcardFrom(ctx context.Context) (logicalcluster.Name, bool, error) {
	cluster, err := ValidClusterFrom(ctx)
	if err != nil {
		return "", false, err
	}
	if cluster.Name.Empty() && !cluster.Wildcard {
		return "", false, buildClusterError("cluster name is empty in the request context", ctx)
	}
	return cluster.Name, cluster.Wildcard, nil
}

// ClusterNameFrom returns a cluster.Name from the value of the cluster key on the ctx.
// If the cluster name is not present or cannot be constructed, then return an error.
func ClusterNameFrom(ctx context.Context) (logicalcluster.Name, error) {
	cluster, err := ValidClusterFrom(ctx)
	if err != nil {
		return "", err
	}
	if cluster.Wildcard {
		return "", buildClusterError("wildcard not supported", ctx)
	}
	if cluster.Name.Empty() {
		return "", buildClusterError("cluster name is empty in the request context", ctx)
	}
	return cluster.Name, nil
}
