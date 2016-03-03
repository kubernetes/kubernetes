// Copyright 2016 The Kubernetes Authors. All rights reserved.
package nanny

import (
	"errors"
	"fmt"

	api "k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_2"
)

type KubernetesClient interface {
	CountNodes() (uint64, error)
	ContainerResources() (*apiv1.ResourceRequirements, error)
	UpdateDeployment(resources *apiv1.ResourceRequirements) error
}

type k8s_client_impl struct {
	namespace, deployment, pod, container string
	clientset                             *release_1_2.Clientset
}

func (k *k8s_client_impl) CountNodes() (uint64, error) {
	opt := api.ListOptions{Watch: false}

	nodes, err := k.clientset.CoreClient.Nodes().List(opt)
	if err != nil {
		return 0, err
	}
	return uint64(len(nodes.Items)), nil
}

func (k *k8s_client_impl) ContainerResources() (*apiv1.ResourceRequirements, error) {
	pod, err := k.clientset.CoreClient.Pods(k.namespace).Get(k.pod)

	if err != nil {
		return nil, err
	}
	for _, container := range pod.Spec.Containers {
		if container.Name != k.container {
			continue
		}
		return &container.Resources, nil
	}
	return nil, errors.New(fmt.Sprintf("Container %s was not found in deployment %s in namespace %s.", k.container, k.deployment, k.namespace))
}

func (k *k8s_client_impl) UpdateDeployment(resources *apiv1.ResourceRequirements) error {
	// First, get the Deployment.
	dep, err := k.clientset.Extensions().Deployments(k.namespace).Get(k.deployment)
	if err != nil {
		return err
	}

	// Modify the Deployment object with our ResourceRequirements.
	for i, container := range dep.Spec.Template.Spec.Containers {
		if container.Name == k.container {
			dep.Spec.Template.Spec.Containers[i].Resources = *resources
			_, err = k.clientset.ExtensionsClient.Deployments(k.namespace).Update(dep)
			return err
		}
	}

	// Update the deployment.
	return errors.New(fmt.Sprintf("Container %s wasn't found in the deployment %d in namespace %d.", k.container, k.deployment, k.namespace))
}

func NewKubernetesClient(namespace, deployment, pod, container string, clientset *release_1_2.Clientset) KubernetesClient {
	return &k8s_client_impl{
		namespace:  namespace,
		deployment: deployment,
		pod:        pod,
		container:  container,
		clientset:  clientset,
	}
}
