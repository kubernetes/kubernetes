//go:build linux

/*
Copyright 2015 The Kubernetes Authors.

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

package nftables

import (
	"context"

	"k8s.io/klog/v2"
	"sigs.k8s.io/knftables"
)

// nftClientBuilder creates a knftables.Interface for cleanup for the given family and table.
type nftClientBuilder func(family knftables.Family, table string) (knftables.Interface, error)

// CleanupLeftovers removes all nftables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func CleanupLeftovers(ctx context.Context) bool {
	return cleanupLeftovers(ctx, func(family knftables.Family, table string) (knftables.Interface, error) {
		return knftables.New(family, table)
	})
}

func cleanupLeftovers(ctx context.Context, newClient nftClientBuilder) bool {
	logger := klog.FromContext(ctx)
	var encounteredError bool

	for _, family := range []knftables.Family{knftables.IPv4Family, knftables.IPv6Family} {
		nft, err := newClient(family, kubeProxyTable)
		if err != nil {
			continue
		}
		tx := nft.NewTransaction()
		tx.Delete(&knftables.Table{})
		err = nft.Run(ctx, tx)
		if err != nil && !knftables.IsNotFound(err) {
			logger.Error(err, "Error cleaning up nftables rules")
			encounteredError = true
		}
	}

	return encounteredError
}
