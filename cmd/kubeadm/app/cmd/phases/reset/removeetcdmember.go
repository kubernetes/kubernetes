/*
Copyright 2019 The Kubernetes Authors.

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

package phases

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	utilstaticpod "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

// NewRemoveETCDMemberPhase creates a kubeadm workflow phase for remove-etcd-member
func NewRemoveETCDMemberPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "remove-etcd-member",
		Short: "Remove a local etcd member.",
		Long:  "Remove a local etcd member for a control plane node.",
		Run:   runRemoveETCDMemberPhase,
		InheritFlags: []string{
			options.KubeconfigPath,
			options.DryRun,
		},
	}
}

func runRemoveETCDMemberPhase(c workflow.RunData) error {
	r, ok := c.(resetData)
	if !ok {
		return errors.New("remove-etcd-member-phase phase invoked with an invalid data struct")
	}
	cfg := r.Cfg()
	var etcdDataDir string

	// Only clear etcd data when using local etcd.
	klog.V(1).Infoln("[reset] Checking for etcd config")
	etcdManifestPath := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ManifestsSubDirName, "etcd.yaml")
	etcdPod, err := getEtcdPod(etcdManifestPath)
	if err != nil {
		return errors.Wrapf(err, "failed to get etcd pod manifest from %s", etcdManifestPath)
	}

	etcdAdvertiseAddress, err := getEtcdAdvertiseAddress(etcdPod)

	if etcdAdvertiseAddress != "" {
		cfg.LocalAPIEndpoint.AdvertiseAddress = etcdAdvertiseAddress
	}

	if cfg != nil && cfg.Etcd.Local != nil {
		etcdDataDir = cfg.Etcd.Local.DataDir
	} else {
		etcdDataDir, err = getEtcdDataDir(etcdPod)
	}
	klog.Warningln("[reset] No kubeadm config, using etcd pod spec to get data directory")
	if err == nil {
		if cfg != nil {
			if !r.DryRun() {
				err := etcdphase.RemoveStackedEtcdMemberFromCluster(r.Client(), cfg)
				if err != nil {
					klog.Warningf("[reset] Failed to remove etcd member: %v, please manually remove this etcd member using etcdctl", err)
				} else {
					if err := CleanDir(etcdDataDir); err != nil {
						klog.Warningf("[reset] Failed to delete contents of the etcd directory: %q, error: %v", etcdDataDir, err)
					} else {
						fmt.Printf("[reset] Deleted contents of the etcd data directory: %v\n", etcdDataDir)
					}
				}
			} else {
				fmt.Println("[reset] Would remove the etcd member on this node from the etcd cluster")
				fmt.Printf("[reset] Would delete contents of the etcd data directory: %v\n", etcdDataDir)
			}
		}
		// This could happen if the phase `cleanup-node` is run before the `remove-etcd-member`.
		// Cleanup the data in the etcd data dir to avoid some stale files which might cause the failure to build cluster in the next time.
		empty, _ := IsDirEmpty(etcdDataDir)
		if !empty && !r.DryRun() {
			if err := CleanDir(etcdDataDir); err != nil {
				klog.Warningf("[reset] Failed to delete contents of the etcd directory: %q, error: %v", etcdDataDir, err)
			} else {
				fmt.Printf("[reset] Deleted contents of the etcd data directory: %v\n", etcdDataDir)
			}
		}
	} else {
		fmt.Println("[reset] No etcd config found. Assuming external etcd")
		fmt.Println("[reset] Please, manually reset etcd to prevent further issues")
	}

	return nil
}

// getEtcdPod returns the etcd pod manifest from the file system
func getEtcdPod(manifestPath string) (*v1.Pod, error) {
	if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
		// Fall back to use the default cluster config if etcd.yaml doesn't exist, this could happen that
		// etcd.yaml is removed by other reset phases, e.g. cleanup-node.
		klog.Infof("[reset] No etcd pod manifest found at %s", manifestPath)

		return nil, nil
	}
	etcdPod, err := utilstaticpod.ReadStaticPodFromDisk(manifestPath)
	if err != nil {
		return nil, err
	}
	return etcdPod, nil
}

// getEtcdDataDir returns the etcd data directory from the etcd pod manifest
func getEtcdDataDir(etcdPod *v1.Pod) (string, error) {
	const etcdVolumeName = "etcd-data"
	var dataDir string

	if etcdPod == nil {
		// Fall back to use the default cluster config if etcd.yaml doesn't exist, this could happen that
		// etcd.yaml is removed by other reset phases, e.g. cleanup-node.
		cfg := &v1beta3.ClusterConfiguration{}
		scheme.Scheme.Default(cfg)
		return cfg.Etcd.Local.DataDir, nil
	}
	for _, volumeMount := range etcdPod.Spec.Volumes {
		if volumeMount.Name == etcdVolumeName {
			dataDir = volumeMount.HostPath.Path
			break
		}
	}
	if dataDir == "" {
		return dataDir, errors.New("invalid etcd pod manifest")
	}
	return dataDir, nil
}

// getEtcdAdvertiseAddress returns the etcd advertise address from the etcd pod manifest
func getEtcdAdvertiseAddress(etcdPod *v1.Pod) (string, error) {
	etcdAdvertiseClientUrls := etcdPod.Annotations[kubeadmconstants.EtcdAdvertiseClientUrlsAnnotationKey]

	if etcdAdvertiseClientUrls != "" {
		u, err := url.Parse(etcdAdvertiseClientUrls)
		if err != nil || (err == nil && u.Hostname() == "") {
			return "", fmt.Errorf("failed to parse etcd advertise client urls: %s", etcdAdvertiseClientUrls)
		}
		return u.Hostname(), nil
	}
	return "", nil
}
