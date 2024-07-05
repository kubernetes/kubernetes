/*
Copyright 2024 The Kubernetes Authors.

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

package endpointslice

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	corelisters "k8s.io/client-go/listers/core/v1"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

// DesiredEndpointSlicesFromServicePods returns the list of desired endpointslices for the given pods and services.
// It also return which address types can be handled by this service.
func DesiredEndpointSlicesFromServicePods(
	logger klog.Logger,
	pods []*corev1.Pod,
	service *corev1.Service,
	nodeLister corelisters.NodeLister,
) ([]*EndpointPortAddressType, sets.Set[discovery.AddressType], error) {
	errs := []error{}
	desiredEndpointSlicesByAddrTypePort := map[endpointsliceutil.PortMapKey]*EndpointPortAddressType{}

	// addresses that this service supports [o(1) find]
	serviceSupportedAddressesTypes := getAddressTypesForService(logger, service)

	for _, pod := range pods {
		if !endpointsliceutil.ShouldPodBeInEndpoints(pod, true) {
			continue
		}

		endpointPorts := getEndpointPorts(logger, service, pod)

		setDesiredEndpointSlicesIfEmpty := func() {
			for addressType := range serviceSupportedAddressesTypes {
				epHash := newAddrTypePortMapKey(endpointsliceutil.NewPortMapKey(endpointPorts), addressType)
				_, exists := desiredEndpointSlicesByAddrTypePort[epHash]
				if !exists {
					desiredEndpointSlicesByAddrTypePort[epHash] = &EndpointPortAddressType{
						EndpointSet: endpointsliceutil.EndpointSet{},
						Ports:       endpointPorts,
						AddressType: addressType,
					}
				}
			}
		}

		node, err := nodeLister.Get(pod.Spec.NodeName)
		if err != nil {
			// we are getting the information from the local informer,
			// an error different than IsNotFound should not happen
			if !errors.IsNotFound(err) {
				setDesiredEndpointSlicesIfEmpty()
				return nil, nil, err
			}
			// If the Node specified by the Pod doesn't exist we want to requeue the Service so we
			// retry later, but also update the EndpointSlice without the problematic Pod.
			// Theoretically, the pod Garbage Collector will remove the Pod, but we want to avoid
			// situations where a reference from a Pod to a missing node can leave the EndpointSlice
			// stuck forever.
			// On the other side, if the service.Spec.PublishNotReadyAddresses is set we just add the
			// Pod, since the user is explicitly indicating that the Pod address should be published.
			if !service.Spec.PublishNotReadyAddresses {
				setDesiredEndpointSlicesIfEmpty()
				logger.Info("skipping Pod for Service, Node not found", "pod", klog.KObj(pod), "service", klog.KObj(service), "node", klog.KRef("", pod.Spec.NodeName))
				errs = append(errs, fmt.Errorf("skipping Pod %s for Service %s/%s: Node %s Not Found", pod.Name, service.Namespace, service.Name, pod.Spec.NodeName))
				continue
			}
		}

		for addressType := range serviceSupportedAddressesTypes {
			endpoint := podToEndpoint(pod, node, service, addressType)
			if len(endpoint.Addresses) == 0 {
				continue
			}

			epHash := newAddrTypePortMapKey(endpointsliceutil.NewPortMapKey(endpointPorts), addressType)
			endpointPortAddrType, exists := desiredEndpointSlicesByAddrTypePort[epHash]
			if !exists {
				endpointPortAddrType = &EndpointPortAddressType{
					EndpointSet: endpointsliceutil.EndpointSet{},
					Ports:       endpointPorts,
					AddressType: addressType,
				}
				desiredEndpointSlicesByAddrTypePort[epHash] = endpointPortAddrType
			}

			endpointPortAddrType.EndpointSet.Insert(&endpoint)
		}
	}

	desiredEndpointSlices := []*EndpointPortAddressType{}
	for _, endpointPortAddrType := range desiredEndpointSlicesByAddrTypePort {
		desiredEndpointSlices = append(desiredEndpointSlices, endpointPortAddrType)
	}

	return desiredEndpointSlices, serviceSupportedAddressesTypes, utilerrors.NewAggregate(errs)
}

type LabelsFromService struct {
	Service *corev1.Service
}

// SetLabels returns a map with the new endpoint slices labels and true if there was an update.
// Slices labels must be equivalent to the Service labels except for the reserved IsHeadlessService, LabelServiceName and LabelManagedBy labels
// Changes to IsHeadlessService, LabelServiceName and LabelManagedBy labels on the Service do not result in updates to EndpointSlice labels.
func (lfs *LabelsFromService) SetLabels(logger klog.Logger, epSlice *discovery.EndpointSlice, controllerName string) (map[string]string, map[string]string, bool) {
	updated := false
	epLabels := make(map[string]string)
	svcLabels := make(map[string]string)

	// check if the endpoint slice and the service have the same labels
	// clone current slice labels except the reserved labels
	for key, value := range epSlice.Labels {
		if isReservedLabelKey(key) {
			continue
		}
		// copy endpoint slice labels
		epLabels[key] = value
	}

	for key, value := range lfs.Service.Labels {
		if isReservedLabelKey(key) {
			logger.Info("Service using reserved endpoint slices label", "service", klog.KObj(lfs.Service), "skipping", key, "label", value)
			continue
		}
		// copy service labels
		svcLabels[key] = value
	}

	// if the labels are not identical update the slice with the corresponding service labels
	if !apiequality.Semantic.DeepEqual(epLabels, svcLabels) {
		updated = true
	}

	// add or remove headless label depending on the service Type
	if !isServiceIPSet(lfs.Service) {
		svcLabels[corev1.IsHeadlessService] = ""
	} else {
		delete(svcLabels, corev1.IsHeadlessService)
	}

	// override endpoint slices reserved labels
	svcLabels[discovery.LabelServiceName] = lfs.Service.Name
	svcLabels[discovery.LabelManagedBy] = controllerName

	return svcLabels, nil, updated
}

// podToEndpoint returns an Endpoint object generated from a Pod, a Node, and a Service for a particular addressType.
func podToEndpoint(pod *corev1.Pod, node *corev1.Node, service *corev1.Service, addressType discovery.AddressType) discovery.Endpoint {
	serving := endpointsliceutil.IsPodReady(pod)
	terminating := pod.DeletionTimestamp != nil
	// For compatibility reasons, "ready" should never be "true" if a pod is terminatng, unless
	// publishNotReadyAddresses was set.
	ready := service.Spec.PublishNotReadyAddresses || (serving && !terminating)
	ep := discovery.Endpoint{
		Addresses: getEndpointAddresses(pod.Status, addressType),
		Conditions: discovery.EndpointConditions{
			Ready:       &ready,
			Serving:     &serving,
			Terminating: &terminating,
		},
		TargetRef: &corev1.ObjectReference{
			Kind:      "Pod",
			Namespace: pod.ObjectMeta.Namespace,
			Name:      pod.ObjectMeta.Name,
			UID:       pod.ObjectMeta.UID,
		},
	}

	if pod.Spec.NodeName != "" {
		ep.NodeName = &pod.Spec.NodeName
	}

	if node != nil && node.Labels[corev1.LabelTopologyZone] != "" {
		zone := node.Labels[corev1.LabelTopologyZone]
		ep.Zone = &zone
	}

	if endpointsliceutil.ShouldSetHostname(pod, service) {
		ep.Hostname = &pod.Spec.Hostname
	}

	return ep
}

// getEndpointPorts returns a list of EndpointPorts generated from a Service
// and Pod.
func getEndpointPorts(logger klog.Logger, service *corev1.Service, pod *corev1.Pod) []discovery.EndpointPort {
	endpointPorts := []discovery.EndpointPort{}

	// Allow headless service not to have ports.
	if len(service.Spec.Ports) == 0 && service.Spec.ClusterIP == corev1.ClusterIPNone {
		return endpointPorts
	}

	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]

		portName := servicePort.Name
		portProto := servicePort.Protocol
		portNum, err := findPort(pod, servicePort)
		if err != nil {
			logger.V(4).Info("Failed to find port for service", "service", klog.KObj(service), "err", err)
			continue
		}

		i32PortNum := int32(portNum)
		endpointPorts = append(endpointPorts, discovery.EndpointPort{
			Name:        &portName,
			Port:        &i32PortNum,
			Protocol:    &portProto,
			AppProtocol: servicePort.AppProtocol,
		})
	}

	return endpointPorts
}

// getEndpointAddresses returns a list of addresses generated from a pod status.
func getEndpointAddresses(podStatus corev1.PodStatus, addressType discovery.AddressType) []string {
	addresses := []string{}

	for _, podIP := range podStatus.PodIPs {
		isIPv6PodIP := netutils.IsIPv6String(podIP.IP)
		if isIPv6PodIP && addressType == discovery.AddressTypeIPv6 {
			addresses = append(addresses, podIP.IP)
		}

		if !isIPv6PodIP && addressType == discovery.AddressTypeIPv4 {
			addresses = append(addresses, podIP.IP)
		}
	}

	return addresses
}

// isReservedLabelKey return true if the label is one of the reserved label for slices
func isReservedLabelKey(label string) bool {
	if label == discovery.LabelServiceName ||
		label == discovery.LabelManagedBy ||
		label == corev1.IsHeadlessService {
		return true
	}
	return false
}

// returns a map of address types used by a service
func getAddressTypesForService(logger klog.Logger, service *corev1.Service) sets.Set[discovery.AddressType] {
	serviceSupportedAddresses := sets.New[discovery.AddressType]()
	// TODO: (khenidak) when address types are removed in favor of
	// v1.IPFamily this will need to be removed, and work directly with
	// v1.IPFamily types

	// IMPORTANT: we assume that IP of (discovery.AddressType enum) is never in use
	// as it gets deprecated
	for _, family := range service.Spec.IPFamilies {
		if family == corev1.IPv4Protocol {
			serviceSupportedAddresses.Insert(discovery.AddressTypeIPv4)
		}

		if family == corev1.IPv6Protocol {
			serviceSupportedAddresses.Insert(discovery.AddressTypeIPv6)
		}
	}

	if serviceSupportedAddresses.Len() > 0 {
		return serviceSupportedAddresses // we have found families for this service
	}

	// TODO (khenidak) remove when (1) dual stack becomes
	// enabled by default (2) v1.19 falls off supported versions

	// Why do we need this:
	// a cluster being upgraded to the new apis
	// will have service.spec.IPFamilies: nil
	// if the controller manager connected to old api
	// server. This will have the nasty side effect of
	// removing all slices already created for this service.
	// this will disable all routing to service vip (ClusterIP)
	// this ensures that this does not happen. Same for headless services
	// we assume it is dual stack, until they get defaulted by *new* api-server
	// this ensures that traffic is not disrupted  until then. But *may*
	// include undesired families for headless services until then.

	if len(service.Spec.ClusterIP) > 0 && service.Spec.ClusterIP != corev1.ClusterIPNone { // headfull
		addrType := discovery.AddressTypeIPv4
		if netutils.IsIPv6String(service.Spec.ClusterIP) {
			addrType = discovery.AddressTypeIPv6
		}
		serviceSupportedAddresses.Insert(addrType)
		logger.V(2).Info("Couldn't find ipfamilies for service. This could happen if controller manager is connected to an old apiserver that does not support ip families yet. EndpointSlices for this Service will use addressType as the IP Family based on familyOf(ClusterIP).", "service", klog.KObj(service), "addressType", addrType, "clusterIP", service.Spec.ClusterIP)
		return serviceSupportedAddresses
	}

	// headless
	// for now we assume two families. This should have minimal side effect
	// if the service is headless with no selector, then this will remain the case
	// if the service is headless with selector then chances are pods are still using single family
	// since kubelet will need to restart in order to start patching pod status with multiple ips
	serviceSupportedAddresses.Insert(discovery.AddressTypeIPv4)
	serviceSupportedAddresses.Insert(discovery.AddressTypeIPv6)
	logger.V(2).Info("Couldn't find ipfamilies for headless service, likely because controller manager is likely connected to an old apiserver that does not support ip families yet. The service endpoint slice will use dual stack families until api-server default it correctly", "service", klog.KObj(service))
	return serviceSupportedAddresses
}

// isServiceIPSet aims to check if the service's ClusterIP is set or not
// the objective is not to perform validation here
// copied from k8s.io/kubernetes/pkg/apis/core/v1/helper
func isServiceIPSet(service *corev1.Service) bool {
	return service.Spec.ClusterIP != corev1.ClusterIPNone && service.Spec.ClusterIP != ""
}

// findPort locates the container port for the given pod and portName.  If the
// targetPort is a number, use that.  If the targetPort is a string, look that
// string up in all named ports in all containers in the target pod.  If no
// match is found, fail.
// copied from k8s.io/kubernetes/pkg/api/v1/pod
func findPort(pod *corev1.Pod, svcPort *corev1.ServicePort) (int, error) {
	portName := svcPort.TargetPort
	switch portName.Type {
	case intstr.String:
		name := portName.StrVal
		for _, container := range pod.Spec.Containers {
			for _, port := range container.Ports {
				if port.Name == name && port.Protocol == svcPort.Protocol {
					return int(port.ContainerPort), nil
				}
			}
		}
	case intstr.Int:
		return portName.IntValue(), nil
	}

	return 0, fmt.Errorf("no suitable port for manifest: %s", pod.UID)
}
