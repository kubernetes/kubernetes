/*
Copyright 2017 The Kubernetes Authors.

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

package selfhosting

import (
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/pkg/api"
)

// CreateSelfHostedControlPlane is responsible for turning a Static Pod-hosted control plane to a self-hosted one
// It achieves that task this way:
// 1. Load the Static Pod specification from disk (from /etc/kubernetes/manifests)
// 2. Extract the PodSpec from that Static Pod specification
// 3. Mutate the PodSpec to be compatible with self-hosting (add the right labels, taints, etc. so it can schedule correctly)
// 4. Build a new DaemonSet object for the self-hosted component in question. Use the above mentioned PodSpec
// 5. Create the DaemonSet resource. Wait until the Pods are running.
// 6. Remove the Static Pod manifest file. The kubelet will stop the original Static Pod-hosted component that was running.
// 7. The self-hosted containers should now step up and take over.
// 8. In order to avoid race conditions, we're still making sure the API /healthz endpoint is healthy
// 9. Do that for the kube-apiserver, kube-controller-manager and kube-scheduler in a loop
func CreateSelfHostedControlPlane(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {

	if err := createTLSSecrets(cfg, client); err != nil {
		return err
	}

	if err := createOpaqueSecrets(cfg, client); err != nil {
		return err
	}

	for _, componentName := range kubeadmconstants.MasterComponents {
		start := time.Now()
		manifestPath := kubeadmconstants.GetStaticPodFilepath(componentName, kubeadmconstants.GetStaticPodDirectory())

		// Load the Static Pod file in order to be able to create a self-hosted variant of that file
		podSpec, err := loadPodSpecFromFile(manifestPath)
		if err != nil {
			return err
		}

		// Build a DaemonSet object from the loaded PodSpec
		ds := buildDaemonSet(cfg, componentName, podSpec)

		// Create the DaemonSet in the API Server
		if err := apiclient.CreateOrUpdateDaemonSet(client, ds); err != nil {
			return err
		}

		// Wait for the self-hosted component to come up
		// TODO: Enforce a timeout
		apiclient.WaitForPodsWithLabel(client, buildSelfHostedWorkloadLabelQuery(componentName))

		// Remove the old Static Pod manifest
		if err := os.RemoveAll(manifestPath); err != nil {
			return fmt.Errorf("unable to delete static pod manifest for %s [%v]", componentName, err)
		}

		// Make sure the API is responsive at /healthz
		// TODO: Follow-up on fixing the race condition here and respect the timeout error that can be returned
		apiclient.WaitForAPI(client)

		fmt.Printf("[self-hosted] self-hosted %s ready after %f seconds\n", componentName, time.Since(start).Seconds())
	}
	return nil
}

// buildDaemonSet is responsible for mutating the PodSpec and return a DaemonSet which is suitable for the self-hosting purporse
func buildDaemonSet(cfg *kubeadmapi.MasterConfiguration, name string, podSpec *v1.PodSpec) *extensions.DaemonSet {
	// Mutate the PodSpec so it's suitable for self-hosting
	mutatePodSpec(cfg, name, podSpec)

	// Return a DaemonSet based on that Spec
	return &extensions.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.AddSelfHostedPrefix(name),
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s-app": kubeadmconstants.AddSelfHostedPrefix(name),
			},
		},
		Spec: extensions.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"k8s-app": kubeadmconstants.AddSelfHostedPrefix(name),
					},
				},
				Spec: *podSpec,
			},
		},
	}
}

// loadPodSpecFromFile reads and decodes a file containing a specification of a Pod
// TODO: Consider using "k8s.io/kubernetes/pkg/volume/util".LoadPodFromFile(filename string) in the future instead.
func loadPodSpecFromFile(manifestPath string) (*v1.PodSpec, error) {
	podBytes, err := ioutil.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}

	staticPod := &v1.Pod{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), podBytes, staticPod); err != nil {
		return nil, fmt.Errorf("unable to decode static pod %v", err)
	}

	return &staticPod.Spec, nil
}

// buildSelfHostedWorkloadLabelQuery creates the right query for matching a self-hosted Pod
func buildSelfHostedWorkloadLabelQuery(componentName string) string {
	return fmt.Sprintf("k8s-app=%s", kubeadmconstants.AddSelfHostedPrefix(componentName))
}
