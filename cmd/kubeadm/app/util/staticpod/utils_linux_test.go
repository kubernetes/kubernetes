//go:build linux
// +build linux

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

package staticpod

import (
	"path/filepath"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

type ownerAndPermissions struct {
	uid         int64
	gid         int64
	permissions uint32
}

func verifyPodSecurityContext(t *testing.T, pod *v1.Pod, wantRunAsUser, wantRunAsGroup int64, wantSupGroup []int64) {
	t.Helper()
	wantPodSecurityContext := &v1.PodSecurityContext{
		RunAsUser:          ptr.To(wantRunAsUser),
		RunAsGroup:         ptr.To(wantRunAsGroup),
		SupplementalGroups: wantSupGroup,
		SeccompProfile: &v1.SeccompProfile{
			Type: v1.SeccompProfileTypeRuntimeDefault,
		},
	}
	if !reflect.DeepEqual(wantPodSecurityContext, pod.Spec.SecurityContext) {
		t.Errorf("unexpected diff in PodSecurityContext,  want: %+v, got: %+v", wantPodSecurityContext, pod.Spec.SecurityContext)
	}
}

func verifyContainerSecurityContext(t *testing.T, container v1.Container, addCaps, dropCaps []v1.Capability, allowPrivielege *bool) {
	t.Helper()
	wantContainerSecurityContext := &v1.SecurityContext{
		AllowPrivilegeEscalation: allowPrivielege,
		Capabilities: &v1.Capabilities{
			Add:  addCaps,
			Drop: dropCaps,
		},
	}
	if !reflect.DeepEqual(wantContainerSecurityContext, container.SecurityContext) {
		t.Errorf("unexpected diff in container SecurityContext, want: %+v, got: %+v", wantContainerSecurityContext, container.SecurityContext)
	}
}

func verifyFilePermissions(t *testing.T, updatedFiles, wantFiles map[string]ownerAndPermissions) {
	t.Helper()
	if !reflect.DeepEqual(updatedFiles, wantFiles) {
		t.Errorf("unexpected diff in file owners and permissions want: %+v, got: %+v", wantFiles, updatedFiles)
	}
}

func TestRunKubeAPIServerAsNonRoot(t *testing.T) {
	cfg := &kubeadm.ClusterConfiguration{}
	pod := ComponentPod(v1.Container{Name: "kube-apiserver"}, nil, nil)
	var runAsUser, runAsGroup, supGroup int64 = 1000, 1001, 1002
	updatedFiles := map[string]ownerAndPermissions{}
	if err := runKubeAPIServerAsNonRoot(&pod, &runAsUser, &runAsGroup, &supGroup, func(path string, uid, gid int64, perms uint32) error {
		updatedFiles[path] = ownerAndPermissions{uid: uid, gid: gid, permissions: perms}
		return nil
	}, cfg); err != nil {
		t.Fatal(err)
	}
	verifyPodSecurityContext(t, &pod, runAsUser, runAsGroup, []int64{supGroup})
	verifyContainerSecurityContext(t, pod.Spec.Containers[0], []v1.Capability{"NET_BIND_SERVICE"}, []v1.Capability{"ALL"}, nil)
	wantUpdateFiles := map[string]ownerAndPermissions{
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPublicKeyName):   {uid: runAsUser, gid: runAsGroup, permissions: 0600},
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPrivateKeyName):  {uid: 0, gid: supGroup, permissions: 0640},
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKeyName):              {uid: runAsUser, gid: runAsGroup, permissions: 0600},
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientKeyName): {uid: runAsUser, gid: runAsGroup, permissions: 0600},
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientKeyName):       {uid: runAsUser, gid: runAsGroup, permissions: 0600},
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerEtcdClientKeyName):    {uid: runAsUser, gid: runAsGroup, permissions: 0600},
	}
	verifyFilePermissions(t, updatedFiles, wantUpdateFiles)
}

func TestRunKubeControllerManagerAsNonRoot(t *testing.T) {
	cfg := &kubeadm.ClusterConfiguration{}
	pod := ComponentPod(v1.Container{Name: "kube-controller-manager"}, nil, nil)
	var runAsUser, runAsGroup, supGroup int64 = 1000, 1001, 1002
	updatedFiles := map[string]ownerAndPermissions{}
	if err := runKubeControllerManagerAsNonRoot(&pod, &runAsUser, &runAsGroup, &supGroup, func(path string, uid, gid int64, perms uint32) error {
		updatedFiles[path] = ownerAndPermissions{uid: uid, gid: gid, permissions: perms}
		return nil
	}, cfg); err != nil {
		t.Fatal(err)
	}
	verifyPodSecurityContext(t, &pod, runAsUser, runAsGroup, []int64{supGroup})
	verifyContainerSecurityContext(t, pod.Spec.Containers[0], nil, []v1.Capability{"ALL"}, ptr.To(false))
	wantUpdateFiles := map[string]ownerAndPermissions{
		filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName): {uid: runAsUser, gid: runAsGroup, permissions: 0600},
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPrivateKeyName):                   {uid: 0, gid: supGroup, permissions: 0640},
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName):                                      {uid: runAsUser, gid: runAsGroup, permissions: 0600},
	}
	verifyFilePermissions(t, updatedFiles, wantUpdateFiles)
}

func TestRunKubeSchedulerAsNonRoot(t *testing.T) {
	pod := ComponentPod(v1.Container{Name: "kube-scheduler"}, nil, nil)
	var runAsUser, runAsGroup int64 = 1000, 1001
	updatedFiles := map[string]ownerAndPermissions{}
	if err := runKubeSchedulerAsNonRoot(&pod, &runAsUser, &runAsGroup, func(path string, uid, gid int64, perms uint32) error {
		updatedFiles[path] = ownerAndPermissions{uid: uid, gid: gid, permissions: perms}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	verifyPodSecurityContext(t, &pod, runAsUser, runAsGroup, nil)
	verifyContainerSecurityContext(t, pod.Spec.Containers[0], nil, []v1.Capability{"ALL"}, ptr.To(false))
	wantUpdateFiles := map[string]ownerAndPermissions{
		filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.SchedulerKubeConfigFileName): {uid: runAsUser, gid: runAsGroup, permissions: 0600},
	}
	verifyFilePermissions(t, updatedFiles, wantUpdateFiles)
}

func TestRunEtcdAsNonRoot(t *testing.T) {
	cfg := &kubeadm.ClusterConfiguration{
		Etcd: kubeadm.Etcd{
			Local: &kubeadm.LocalEtcd{
				DataDir: "/var/lib/etcd/data",
			},
		},
	}
	pod := ComponentPod(v1.Container{Name: "etcd"}, nil, nil)
	var runAsUser, runAsGroup int64 = 1000, 1001
	updatedFiles := map[string]ownerAndPermissions{}
	if err := runEtcdAsNonRoot(&pod, &runAsUser, &runAsGroup, func(path string, uid, gid int64, perms uint32) error {
		updatedFiles[path] = ownerAndPermissions{uid: uid, gid: gid, permissions: perms}
		return nil
	},
		func(path string, uid, gid int64) error {
			updatedFiles[path] = ownerAndPermissions{uid: uid, gid: gid, permissions: 0700}
			return nil
		}, cfg); err != nil {
		t.Fatal(err)
	}
	verifyPodSecurityContext(t, &pod, runAsUser, runAsGroup, nil)
	verifyContainerSecurityContext(t, pod.Spec.Containers[0], nil, []v1.Capability{"ALL"}, ptr.To(false))
	wantUpdateFiles := map[string]ownerAndPermissions{
		cfg.Etcd.Local.DataDir: {uid: runAsUser, gid: runAsGroup, permissions: 0700},
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdServerKeyName): {uid: runAsUser, gid: runAsGroup, permissions: 0600},
		filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdPeerKeyName):   {uid: runAsUser, gid: runAsGroup, permissions: 0600},
	}
	verifyFilePermissions(t, updatedFiles, wantUpdateFiles)
}
