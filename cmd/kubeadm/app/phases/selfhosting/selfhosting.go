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
	"path/filepath"
	"time"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
)

const (
	kubeAPIServer         = "kube-apiserver"
	kubeControllerManager = "kube-controller-manager"
	kubeScheduler         = "kube-scheduler"

	selfHostingPrefix = "self-hosted-"
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
func CreateSelfHostedControlPlane(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset) error {

	if err := createTLSSecrets(cfg, client); err != nil {
		return err
	}

	if err := createOpaqueSecrets(cfg, client); err != nil {
		return err
	}

	// The sequence here isn't set in stone, but seems to work well to start with the API server
	components := []string{kubeAPIServer, kubeControllerManager, kubeScheduler}

	for _, componentName := range components {
		start := time.Now()
		manifestPath := buildStaticManifestFilepath(componentName)

		// Load the Static Pod file in order to be able to create a self-hosted variant of that file
		podSpec, err := loadPodSpecFromFile(manifestPath)
		if err != nil {
			return err
		}

		// Build a DaemonSet object from the loaded PodSpec
		ds := buildDaemonSet(cfg, componentName, podSpec)

		// Create the DaemonSet in the API Server
		if _, err := client.ExtensionsV1beta1().DaemonSets(metav1.NamespaceSystem).Create(ds); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("failed to create self-hosted %q daemonset [%v]", componentName, err)
			}

			if _, err := client.ExtensionsV1beta1().DaemonSets(metav1.NamespaceSystem).Update(ds); err != nil {
				// TODO: We should retry on 409 responses
				return fmt.Errorf("failed to update self-hosted %q daemonset [%v]", componentName, err)
			}
		}

		// Wait for the self-hosted component to come up
		kubeadmutil.WaitForPodsWithLabel(client, buildSelfHostedWorkloadLabelQuery(componentName))

		// Remove the old Static Pod manifest
		if err := os.RemoveAll(manifestPath); err != nil {
			return fmt.Errorf("unable to delete static pod manifest for %s [%v]", componentName, err)
		}

		// Make sure the API is responsive at /healthz
		kubeadmutil.WaitForAPI(client)

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
			Name:      addSelfHostedPrefix(name),
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s-app": addSelfHostedPrefix(name),
			},
		},
		Spec: extensions.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"k8s-app": addSelfHostedPrefix(name),
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

// buildStaticManifestFilepath returns the location on the disk where the Static Pod should be present
func buildStaticManifestFilepath(componentName string) string {
	return filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ManifestsSubDirName, componentName+".yaml")
}

// buildSelfHostedWorkloadLabelQuery creates the right query for matching a self-hosted Pod
func buildSelfHostedWorkloadLabelQuery(componentName string) string {
	return fmt.Sprintf("k8s-app=%s", addSelfHostedPrefix(componentName))
}

// addSelfHostedPrefix adds the self-hosted- prefix to the component name
func addSelfHostedPrefix(componentName string) string {
	return fmt.Sprintf("%s%s", selfHostingPrefix, componentName)
}
