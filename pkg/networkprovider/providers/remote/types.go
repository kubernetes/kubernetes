/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package remote

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/networkprovider"
)

type Response struct {
	Err string `json:"Err"`
}

func (r *Response) GetError() string {
	return r.Err
}

type ActivateResponse struct {
	Result bool `json:"Result"`
	Response
}

type CheckTenantIDRequest struct {
	TenantID string `json:"tenantID,omitempty"`
}

type CheckTenantIDResponse struct {
	Result bool `json:"Result"`
	Response
}

type GetNetworkRequest struct {
	Name string `json:"name,omitempty"`
	ID   string `json:"id,omitempty"`
}

type GetNetworkResponse struct {
	Response
	Result *networkprovider.Network `json:"Result"`
}

type CreateNetworkRequest struct {
	Network *networkprovider.Network `json:"Network,omitempty"`
}

type CreateNetworkResponse struct {
	Response
}

type UpdateNetworkRequest struct {
	Network *networkprovider.Network `json:"Network,omitempty"`
}

type UpdateNetworkResponse struct {
	Response
}

type DeleteNetworkRequest struct {
	Name string `json:"name"`
}

type DeleteNetworkResponse struct {
	Response
}

type GetLoadBalancerRequest struct {
	Name string `json:"name"`
}

type GetLoadBalancerResponse struct {
	Response
	Result *networkprovider.LoadBalancer `json:"Result"`
}

type CreateLoadBalancerRequest struct {
	LoadBalancer *networkprovider.LoadBalancer `json:"loadBalancer"`
	Affinity     api.ServiceAffinity           `json:"affinity"`
}

type CreateLoadBalancerResult struct {
	VIP string `json:"VIP"`
}

type CreateLoadBalancerResponse struct {
	Result *CreateLoadBalancerResult `json:"Result"`
	Response
}

type UpdateLoadBalancerRequest struct {
	Name        string                      `json:"name"`
	Hosts       []*networkprovider.HostPort `json:"hosts"`
	ExternalIPs []string                    `json:"externalIPs"`
}

type UpdateLoadBalancerResult struct {
	VIP string `json:"VIP,omitempty"`
}

type UpdateLoadBalancerResponse struct {
	Result *UpdateLoadBalancerResult `json:"Result"`
	Response
}

type DeleteLoadBalancerRequest struct {
	Name string `json:"name"`
}

type DeleteLoadBalancerResponse struct {
	Response
}

type SetupPodRequest struct {
	PodName             string                   `json:"podName"`
	Namespace           string                   `json:"namespace"`
	ContainerRuntime    string                   `json:"containerRuntime"`
	PodInfraContainerID string                   `json:"podInfraContainerID"`
	Network             *networkprovider.Network `json:"network"`
}

type SetupPodResponse struct {
	Response
}

type TeardownPodRequest struct {
	PodName             string                   `json:"podName"`
	Namespace           string                   `json:"namespace"`
	ContainerRuntime    string                   `json:"containerRuntime"`
	PodInfraContainerID string                   `json:"podInfraContainerID"`
	Network             *networkprovider.Network `json:"network"`
}

type TeardownPodResponse struct {
	Response
}

type PodStatusRequest struct {
	PodName             string                   `json:"podName"`
	Namespace           string                   `json:"namespace"`
	ContainerRuntime    string                   `json:"containerRuntime"`
	PodInfraContainerID string                   `json:"podInfraContainerID"`
	Network             *networkprovider.Network `json:"network"`
}

type PodStatusResult struct {
	IP string `json:"IP,omitempty"`
}

type PodStatusResponse struct {
	Result *PodStatusResult `json:"Result"`
	Response
}
