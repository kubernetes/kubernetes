/*
Copyright 2022 The KCP Authors.

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

package logicalcluster

import "context"

type key int

const (
	keyCluster key = iota
)

// WithCluster injects a cluster name into a context
func WithCluster(ctx context.Context, cluster Name) context.Context {
	return context.WithValue(ctx, keyCluster, cluster)
}

// ClusterFromContext extracts a cluster name from the context
func ClusterFromContext(ctx context.Context) (Name, bool) {
	s, ok := ctx.Value(keyCluster).(Name)
	return s, ok
}
