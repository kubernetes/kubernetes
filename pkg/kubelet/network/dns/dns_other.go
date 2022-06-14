//go:build !windows
// +build !windows

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

package dns

import (
	"fmt"
	"os"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
)

func getHostDNSConfig(resolverConfig string) (*runtimeapi.DNSConfig, error) {
	var hostDNS, hostSearch, hostOptions []string
	// Get host DNS settings
	if resolverConfig != "" {
		f, err := os.Open(resolverConfig)
		if err != nil {
			klog.ErrorS(err, "Could not open resolv conf file.")
			return nil, err
		}
		defer f.Close()

		hostDNS, hostSearch, hostOptions, err = parseResolvConf(f)
		if err != nil {
			err := fmt.Errorf("Encountered error while parsing resolv conf file. Error: %w", err)
			klog.ErrorS(err, "Could not parse resolv conf file.")
			return nil, err
		}
	}
	return &runtimeapi.DNSConfig{
		Servers:  hostDNS,
		Searches: hostSearch,
		Options:  hostOptions,
	}, nil
}
