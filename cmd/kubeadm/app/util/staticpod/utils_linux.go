//go:build linux

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

// Package staticpod contains utilities for managing the static pod.
package staticpod

import (
	"fmt"
	"path/filepath"

	v1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/users"
)

type pathOwnerAndPermissionsUpdaterFunc func(path string, uid, gid int64, perms uint32) error
type pathOwnerUpdaterFunc func(path string, uid, gid int64) error

// RunComponentAsNonRoot updates the pod manifest and the hostVolume permissions to run as non root.
func RunComponentAsNonRoot(componentName string, pod *v1.Pod, usersAndGroups *users.UsersAndGroups, cfg *kubeadmapi.ClusterConfiguration) error {
	switch componentName {
	case kubeadmconstants.KubeAPIServer:
		return runKubeAPIServerAsNonRoot(
			pod,
			usersAndGroups.Users.ID(kubeadmconstants.KubeAPIServerUserName),
			usersAndGroups.Groups.ID(kubeadmconstants.KubeAPIServerUserName),
			usersAndGroups.Groups.ID(kubeadmconstants.ServiceAccountKeyReadersGroupName),
			users.UpdatePathOwnerAndPermissions,
			cfg,
		)
	case kubeadmconstants.KubeControllerManager:
		return runKubeControllerManagerAsNonRoot(
			pod,
			usersAndGroups.Users.ID(kubeadmconstants.KubeControllerManagerUserName),
			usersAndGroups.Groups.ID(kubeadmconstants.KubeControllerManagerUserName),
			usersAndGroups.Groups.ID(kubeadmconstants.ServiceAccountKeyReadersGroupName),
			users.UpdatePathOwnerAndPermissions,
			cfg,
		)
	case kubeadmconstants.KubeScheduler:
		return runKubeSchedulerAsNonRoot(
			pod,
			usersAndGroups.Users.ID(kubeadmconstants.KubeSchedulerUserName),
			usersAndGroups.Groups.ID(kubeadmconstants.KubeSchedulerUserName),
			users.UpdatePathOwnerAndPermissions,
		)
	case kubeadmconstants.Etcd:
		return runEtcdAsNonRoot(
			pod,
			usersAndGroups.Users.ID(kubeadmconstants.EtcdUserName),
			usersAndGroups.Groups.ID(kubeadmconstants.EtcdUserName),
			users.UpdatePathOwnerAndPermissions,
			users.UpdatePathOwner,
			cfg,
		)
	}
	return errors.New(fmt.Sprintf("component name %q is not valid", componentName))
}

// runKubeAPIServerAsNonRoot updates the pod manifest and the hostVolume permissions to run kube-apiserver as non root.
func runKubeAPIServerAsNonRoot(pod *v1.Pod, runAsUser, runAsGroup, supplementalGroup *int64, updatePathOwnerAndPermissions pathOwnerAndPermissionsUpdaterFunc, cfg *kubeadmapi.ClusterConfiguration) error {
	saPublicKeyFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPublicKeyName)
	if err := updatePathOwnerAndPermissions(saPublicKeyFile, *runAsUser, *runAsGroup, 0600); err != nil {
		return err
	}
	saPrivateKeyFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPrivateKeyName)
	if err := updatePathOwnerAndPermissions(saPrivateKeyFile, 0, *supplementalGroup, 0640); err != nil {
		return err
	}
	apiServerKeyFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKeyName)
	if err := updatePathOwnerAndPermissions(apiServerKeyFile, *runAsUser, *runAsGroup, 0600); err != nil {
		return err
	}
	apiServerKubeletClientKeyFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientKeyName)
	if err := updatePathOwnerAndPermissions(apiServerKubeletClientKeyFile, *runAsUser, *runAsGroup, 0600); err != nil {
		return err
	}
	frontProxyClientKeyName := filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientKeyName)
	if err := updatePathOwnerAndPermissions(frontProxyClientKeyName, *runAsUser, *runAsGroup, 0600); err != nil {
		return err
	}
	if cfg.Etcd.External == nil {
		apiServerEtcdClientKeyFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerEtcdClientKeyName)
		if err := updatePathOwnerAndPermissions(apiServerEtcdClientKeyFile, *runAsUser, *runAsGroup, 0600); err != nil {
			return err
		}
	}
	pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
		Capabilities: &v1.Capabilities{
			// We drop all capabilities that are added by default.
			Drop: []v1.Capability{"ALL"},
			// kube-apiserver binary has the file capability cap_net_bind_service applied to it.
			// This means that we must add this capability when running as non-root even if the
			// capability is not required.
			Add: []v1.Capability{"NET_BIND_SERVICE"},
		},
	}
	pod.Spec.SecurityContext.RunAsGroup = runAsGroup
	pod.Spec.SecurityContext.RunAsUser = runAsUser
	pod.Spec.SecurityContext.SupplementalGroups = []int64{*supplementalGroup}
	return nil
}

// runKubeControllerManagerAsNonRoot updates the pod manifest and the hostVolume permissions to run kube-controller-manager as non root.
func runKubeControllerManagerAsNonRoot(pod *v1.Pod, runAsUser, runAsGroup, supplementalGroup *int64, updatePathOwnerAndPermissions pathOwnerAndPermissionsUpdaterFunc, cfg *kubeadmapi.ClusterConfiguration) error {
	kubeconfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName)
	if err := updatePathOwnerAndPermissions(kubeconfigFile, *runAsUser, *runAsGroup, 0600); err != nil {
		return err
	}
	saPrivateKeyFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPrivateKeyName)
	if err := updatePathOwnerAndPermissions(saPrivateKeyFile, 0, *supplementalGroup, 0640); err != nil {
		return err
	}
	if res, _ := certphase.UsingExternalCA(cfg); !res {
		caKeyFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName)
		err := updatePathOwnerAndPermissions(caKeyFile, *runAsUser, *runAsGroup, 0600)
		if err != nil {
			return err
		}
	}
	pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
		AllowPrivilegeEscalation: ptr.To(false),
		Capabilities: &v1.Capabilities{
			// We drop all capabilities that are added by default.
			Drop: []v1.Capability{"ALL"},
		},
	}
	pod.Spec.SecurityContext.RunAsUser = runAsUser
	pod.Spec.SecurityContext.RunAsGroup = runAsGroup
	pod.Spec.SecurityContext.SupplementalGroups = []int64{*supplementalGroup}
	return nil
}

// runKubeSchedulerAsNonRoot updates the pod manifest and the hostVolume permissions to run kube-scheduler as non root.
func runKubeSchedulerAsNonRoot(pod *v1.Pod, runAsUser, runAsGroup *int64, updatePathOwnerAndPermissions pathOwnerAndPermissionsUpdaterFunc) error {
	kubeconfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.SchedulerKubeConfigFileName)
	if err := updatePathOwnerAndPermissions(kubeconfigFile, *runAsUser, *runAsGroup, 0600); err != nil {
		return err
	}
	pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
		AllowPrivilegeEscalation: ptr.To(false),
		// We drop all capabilities that are added by default.
		Capabilities: &v1.Capabilities{
			Drop: []v1.Capability{"ALL"},
		},
	}
	pod.Spec.SecurityContext.RunAsUser = runAsUser
	pod.Spec.SecurityContext.RunAsGroup = runAsGroup
	return nil
}

// runEtcdAsNonRoot updates the pod manifest and the hostVolume permissions to run etcd as non root.
func runEtcdAsNonRoot(pod *v1.Pod, runAsUser, runAsGroup *int64, updatePathOwnerAndPermissions pathOwnerAndPermissionsUpdaterFunc, updatePathOwner pathOwnerUpdaterFunc, cfg *kubeadmapi.ClusterConfiguration) error {
	if err := updatePathOwner(cfg.Etcd.Local.DataDir, *runAsUser, *runAsGroup); err != nil {
		return err
	}
	etcdServerKeyFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdServerKeyName)
	if err := updatePathOwnerAndPermissions(etcdServerKeyFile, *runAsUser, *runAsGroup, 0600); err != nil {
		return err
	}
	etcdPeerKeyFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdPeerKeyName)
	if err := updatePathOwnerAndPermissions(etcdPeerKeyFile, *runAsUser, *runAsGroup, 0600); err != nil {
		return err
	}
	pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
		AllowPrivilegeEscalation: ptr.To(false),
		// We drop all capabilities that are added by default.
		Capabilities: &v1.Capabilities{
			Drop: []v1.Capability{"ALL"},
		},
	}
	pod.Spec.SecurityContext.RunAsUser = runAsUser
	pod.Spec.SecurityContext.RunAsGroup = runAsGroup
	return nil
}
