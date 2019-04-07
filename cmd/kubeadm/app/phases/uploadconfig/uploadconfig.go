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

package uploadconfig

import (
	"context"
	"fmt"
	"time"

	"github.com/pkg/errors"
	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

type updateKubeadmConfigMap func() error

const (
	// NodesKubeadmConfigClusterRoleName sets the name for the ClusterRole that allows
	// the bootstrap tokens to access the kubeadm-config ConfigMap during the node bootstrap/discovery
	// or during upgrade nodes
	NodesKubeadmConfigClusterRoleName = "kubeadm:nodes-kubeadm-config"
)

var (
	lockIdentity = string(uuid.NewUUID())
)

// ResetClusterStatusForNode removes the APIEndpoint of a given control-plane node
// from the ClusterStatus and updates the kubeadm ConfigMap
func ResetClusterStatusForNode(cfg *kubeadmapi.InitConfiguration, client clientset.Interface) error {
	fmt.Printf("[reset] Removing info for node %q from the ConfigMap %q in the %q Namespace\n",
		cfg.NodeRegistration.Name, kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)

	return exclusivelyUpdateKubeadmConfigMap(client, func() error {
		return removeCurrentEndpointFromClusterConfiguration(cfg, client)
	})
}

func exclusivelyUpdateKubeadmConfigMap(client clientset.Interface, updateConfigMap updateKubeadmConfigMap) error {
	lock, err := resourcelock.New(resourcelock.ConfigMapsResourceLock, metav1.NamespaceSystem, kubeadmconstants.KubeadmConfigConfigMap, client.CoreV1(), nil, resourcelock.ResourceLockConfig{
		Identity: lockIdentity,
	})
	if err != nil {
		return errors.Wrap(err, "failed to create a configmap resource lock")
	}

	ctx, cancel := context.WithCancel(context.TODO())
	res := make(chan error, 1)

	le, err := leaderelection.NewLeaderElector(leaderelection.LeaderElectionConfig{
		Lock:            lock,
		LeaseDuration:   15 * time.Second,
		RenewDeadline:   10 * time.Second,
		RetryPeriod:     5 * time.Second,
		ReleaseOnCancel: true,
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: func(context.Context) {
				res <- updateConfigMap()
				// Before trying to release the lock by cancelling the leader election context,
				// make sure it's refreshed with the latest contents, avoiding patch conflicts
				lock.Get()
				cancel()
			},
		},
	})
	if err != nil {
		return errors.Wrap(err, "failed to create a leader elector")
	}

	le.Run(ctx)

	return <-res
}

func addCurrentEndpointToClusterConfiguration(cfg *kubeadmapi.InitConfiguration, client clientset.Interface) error {
	// Prepare the ClusterConfiguration for upload
	// The components store their config in their own ConfigMaps, then reset the .ComponentConfig struct;
	// We don't want to mutate the cfg itself, so create a copy of it using .DeepCopy of it first
	clusterConfigurationToUpload := cfg.ClusterConfiguration.DeepCopy()
	clusterConfigurationToUpload.ComponentConfigs = kubeadmapi.ComponentConfigs{}

	// Marshal the ClusterConfiguration into YAML
	clusterConfigurationYaml, err := configutil.MarshalKubeadmConfigObject(clusterConfigurationToUpload)

	clusterStatus, err := configutil.GetClusterStatus(client)
	if err != nil {
		clusterStatus = &kubeadmapi.ClusterStatus{}
	}

	// Updates the ClusterStatus with the current control plane instance
	if clusterStatus.APIEndpoints == nil {
		clusterStatus.APIEndpoints = map[string]kubeadmapi.APIEndpoint{}
	}
	clusterStatus.APIEndpoints[cfg.NodeRegistration.Name] = cfg.LocalAPIEndpoint

	// Marshal the ClusterStatus back into YAML
	clusterStatusYaml, err := configutil.MarshalKubeadmConfigObject(clusterStatus)
	if err != nil {
		return err
	}

	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeadmConfigConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.ClusterConfigurationConfigMapKey: string(clusterConfigurationYaml),
			kubeadmconstants.ClusterStatusConfigMapKey:        string(clusterStatusYaml),
		},
	}

	// Update the ClusterStatus in the ConfigMap
	return apiclient.CreateOrPatchConfigMap(client, configMap)
}

func removeCurrentEndpointFromClusterConfiguration(cfg *kubeadmapi.InitConfiguration, client clientset.Interface) error {
	clusterStatus, err := configutil.GetClusterStatus(client)
	if err != nil {
		klog.Warning("Could not retrieve cluster status")
		return nil
	}

	// Handle a nil APIEndpoints map. Should only happen if someone manually
	// interacted with the ConfigMap.
	if clusterStatus.APIEndpoints == nil {
		return errors.Errorf("APIEndpoints from ConfigMap %q in the %q Namespace is nil",
			kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
	}

	// Check for existence of the nodeName key in the list of APIEndpoints.
	// Return early if it's missing.
	apiEndpoint, ok := clusterStatus.APIEndpoints[cfg.NodeRegistration.Name]
	if !ok {
		klog.Warningf("No APIEndpoint registered for node %q", cfg.NodeRegistration.Name)
		return nil
	}

	klog.V(2).Infof("Removing APIEndpoint %#v for node %q", apiEndpoint, cfg.NodeRegistration.Name)
	delete(clusterStatus.APIEndpoints, cfg.NodeRegistration.Name)

	// Marshal the ClusterStatus back into YAML
	clusterStatusYaml, err := configutil.MarshalKubeadmConfigObject(clusterStatus)
	if err != nil {
		return err
	}

	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeadmConfigConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.ClusterStatusConfigMapKey: string(clusterStatusYaml),
		},
	}

	// Update the ClusterStatus in the ConfigMap
	return apiclient.CreateOrPatchConfigMap(client, configMap)
}

// UploadConfiguration saves the InitConfiguration used for later reference (when upgrading for instance)
func UploadConfiguration(cfg *kubeadmapi.InitConfiguration, client clientset.Interface) error {
	fmt.Printf("[upload-config] Storing the configuration used in ConfigMap %q in the %q Namespace\n", kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)

	err := exclusivelyUpdateKubeadmConfigMap(client, func() error {
		return addCurrentEndpointToClusterConfiguration(cfg, client)
	})
	if err != nil {
		return err
	}

	// Ensure that the NodesKubeadmConfigClusterRoleName exists
	err = apiclient.CreateOrUpdateRole(client, &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      NodesKubeadmConfigClusterRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		Rules: []rbac.PolicyRule{
			rbachelper.NewRule("get").Groups("").Resources("configmaps").Names(kubeadmconstants.KubeadmConfigConfigMap).RuleOrDie(),
		},
	})
	if err != nil {
		return err
	}

	// Binds the NodesKubeadmConfigClusterRoleName to all the bootstrap tokens
	// that are members of the system:bootstrappers:kubeadm:default-node-token group
	// and to all nodes
	return apiclient.CreateOrUpdateRoleBinding(client, &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      NodesKubeadmConfigClusterRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     NodesKubeadmConfigClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: rbac.GroupKind,
				Name: kubeadmconstants.NodeBootstrapTokenAuthGroup,
			},
			{
				Kind: rbac.GroupKind,
				Name: kubeadmconstants.NodesGroup,
			},
		},
	})
}
