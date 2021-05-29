/*
Copyright 2021 The Kubernetes Authors.

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

package v1beta1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/api/discovery/v1beta1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/discovery"
)

func Convert_v1beta1_Endpoint_To_discovery_Endpoint(in *v1beta1.Endpoint, out *discovery.Endpoint, s conversion.Scope) error {
	if err := autoConvert_v1beta1_Endpoint_To_discovery_Endpoint(in, out, s); err != nil {
		return err
	}
	if in.Topology != nil {
		// Copy	Topology into Deprecated Topology
		out.DeprecatedTopology = make(map[string]string, len(in.Topology))
		for k, v := range in.Topology {
			out.DeprecatedTopology[k] = v
		}

		// Move zone from the topology map into a field
		if zone, ok := in.Topology[corev1.LabelTopologyZone]; ok {
			out.Zone = &zone
			delete(out.DeprecatedTopology, corev1.LabelTopologyZone)
		}

		// Remove hostname from the topology map ONLY IF it is the same value as
		// nodeName.  This preserves the (rather odd) ability to have different
		// values for topology[hostname] and nodename in v1beta1, without showing
		// duplicate values in v1.
		if node, ok := in.Topology[corev1.LabelHostname]; ok {
			if out.NodeName != nil && node == *out.NodeName {
				delete(out.DeprecatedTopology, corev1.LabelHostname)
			}
		}

		// If zone & hostname were the only field in the map or topology was empty
		// set DeprecatedTopology to nil
		if len(out.DeprecatedTopology) == 0 {
			out.DeprecatedTopology = nil
		}
	}

	return nil
}

func Convert_discovery_Endpoint_To_v1beta1_Endpoint(in *discovery.Endpoint, out *v1beta1.Endpoint, s conversion.Scope) error {
	if err := autoConvert_discovery_Endpoint_To_v1beta1_Endpoint(in, out, s); err != nil {
		return err
	}

	// If no deprecated topology, zone or node field, no conversion is necessary
	if in.DeprecatedTopology == nil && in.Zone == nil && in.NodeName == nil {
		return nil
	}

	// Copy	Deprecated Topology into Topology
	out.Topology = make(map[string]string, len(in.DeprecatedTopology))
	for k, v := range in.DeprecatedTopology {
		out.Topology[k] = v
	}

	// Add zone field into the topology map
	if in.Zone != nil {
		out.Topology[corev1.LabelTopologyZone] = *in.Zone
	}

	// Add hostname into the topology map ONLY IF it is not already present.
	// This preserves the (rather odd) ability to have different values for
	// topology[hostname] and nodename in v1beta1.
	if in.NodeName != nil && out.Topology[corev1.LabelHostname] == "" {
		out.Topology[corev1.LabelHostname] = *in.NodeName
	}

	return nil
}
