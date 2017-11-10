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

package etcd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"

	"k8s.io/api/core/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

const (
	etcdVolumeName    = "etcd"
	etcdPkiVolumeName = "pki"
)

// CreateLocalEtcdStaticPodManifestFile will write local etcd static pod manifest file.
func CreateLocalEtcdStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.MasterConfiguration) error {

	// gets etcd StaticPodSpec, actualized for the current MasterConfiguration
	spec := GetEtcdPodSpec(cfg)

	// writes etcd StaticPod to disk
	if err := staticpodutil.WriteStaticPodToDisk(kubeadmconstants.Etcd, manifestDir, spec); err != nil {
		return err
	}

	fmt.Printf("[etcd] Wrote Static Pod manifest for a local etcd instance to %q\n", kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.Etcd, manifestDir))
	return nil
}

// GetEtcdPodSpec returns the etcd static Pod actualized to the context of the current MasterConfiguration
// NB. GetEtcdPodSpec methods holds the information about how kubeadm creates etcd static pod mainfests.
func GetEtcdPodSpec(cfg *kubeadmapi.MasterConfiguration) v1.Pod {
	pathType := v1.HostPathDirectoryOrCreate
	etcdMounts := map[string]v1.Volume{
		etcdVolumeName: staticpodutil.NewVolume(etcdVolumeName, cfg.Etcd.DataDir, &pathType),
	}

	clientPort, peerPort := GetEtcdPorts(cfg)
	return staticpodutil.ComponentPod(v1.Container{
		Name:    kubeadmconstants.Etcd,
		Command: getEtcdCommand(cfg, clientPort, peerPort),
		Image:   images.GetCoreImage(kubeadmconstants.Etcd, cfg.ImageRepository, "", cfg.Etcd.Image),
		// Mount the etcd datadir path read-write so etcd can store data in a more persistent manner
		VolumeMounts:  []v1.VolumeMount{staticpodutil.NewVolumeMount(etcdVolumeName, cfg.Etcd.DataDir, false)},
		LivenessProbe: staticpodutil.ComponentProbe(cfg, kubeadmconstants.Etcd, clientPort, "/health", v1.URISchemeHTTP),
	}, etcdMounts)
}

// GetEtcdPorts retrieves the relevant ports for the bootstrap etcd cluster (static pods)
// Normally these are 2379/2380, but if HA is enabled, they will use 12379/12380 because
// the self-hosted etcd cluster will claim these ports instead
func GetEtcdPorts(cfg *kubeadmapi.MasterConfiguration) (int, int) {
	clientPort, peerPort := 2379, 2380
	if features.Enabled(cfg.FeatureGates, features.HighAvailability) {
		clientPort, peerPort = 12379, 12380
	}
	return clientPort, peerPort
}

// getEtcdCommand builds the right etcd command from the given config object
func getEtcdCommand(cfg *kubeadmapi.MasterConfiguration, clientPort, peerPort int) []string {
	defaultArguments := map[string]string{
		"listen-peer-urls":            fmt.Sprintf("http://127.0.0.1:%d", peerPort),
		"initial-advertise-peer-urls": fmt.Sprintf("http://127.0.0.1:%d", peerPort),
		"listen-client-urls":          fmt.Sprintf("http://127.0.0.1:%d", clientPort),
		"advertise-client-urls":       fmt.Sprintf("http://127.0.0.1:%d", clientPort),
		"initial-cluster":             fmt.Sprintf("%s=http://127.0.0.1:%d", kubeadmconstants.EtcdBootstrapMemberName, peerPort),
		"initial-cluster-state":       "new",
		"name":                        kubeadmconstants.EtcdBootstrapMemberName,
		"data-dir":                    cfg.Etcd.DataDir,
	}

	command := []string{"etcd"}
	command = append(command, kubeadmutil.BuildArgumentListFromMap(defaultArguments, cfg.Etcd.ExtraArgs)...)
	return command
}

// DeleteBootstrapEtcd will first retrieve the ID of the bootstrap member then delete it from the cluster via
// the etcd API. Once this done, the manifest of the static pod will be deleted (which will subsequently delete
// the mirror pod) and the data dir is also deleted.
func DeleteBootstrapEtcd(manifestsDir string, cfg *kubeadmapi.MasterConfiguration, waiter apiclient.Waiter) error {
	start := time.Now()

	componentName := "etcd"
	manifestPath := kubeadmconstants.GetStaticPodFilepath(componentName, manifestsDir)

	// set up etcd client so it can connect to the self-hosted etcd cluster
	etcdTLSInfo, err := transport.TLSInfo{
		TrustedCAFile: filepath.Join(cfg.Etcd.SelfHosted.CertificatesDir, kubeadmconstants.CACertName),
		CertFile:      filepath.Join(cfg.Etcd.SelfHosted.CertificatesDir, kubeadmconstants.EtcdClientCertName),
		KeyFile:       filepath.Join(cfg.Etcd.SelfHosted.CertificatesDir, kubeadmconstants.EtcdClientKeyName),
	}.ClientConfig()
	if err != nil {
		return err
	}
	clientCfg := clientv3.Config{
		Endpoints:   []string{"https://localhost:2379"},
		DialTimeout: 5 * time.Second,
		TLS:         etcdTLSInfo,
	}
	etcdcli, err := clientv3.New(clientCfg)
	if err != nil {
		return fmt.Errorf("[self-hosted] Error setting up etcd client: %v", err)
	}
	defer etcdcli.Close()

	fmt.Println("[self-hosted] Listing over etcd cluster to find bootstrap ID")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	memberList, err := etcdcli.MemberList(ctx)
	if err != nil {
		return err
	}

	var bootstrapID uint64
	for _, member := range memberList.Members {
		if member.Name == kubeadmconstants.EtcdBootstrapMemberName {
			bootstrapID = member.ID
		}
	}
	if bootstrapID == 0 {
		return errors.New("[self-hosted] Could not find bootstrap etcd ID")
	}

	_, err = etcdcli.Cluster.MemberRemove(ctx, bootstrapID)
	cancel()
	if err != nil {
		return err
	}

	fmt.Printf("[self-hosted] Removed %d from etcd cluster\n", bootstrapID)
	time.Sleep(2 * time.Second)

	// Remove the old Static Pod manifest
	if err := os.RemoveAll(manifestPath); err != nil {
		return fmt.Errorf("[self-hosted] unable to delete static pod manifest for %s [%v]", componentName, err)
	}

	// Remove data dir too
	if err := os.RemoveAll(cfg.Etcd.DataDir); err != nil {
		return fmt.Errorf("[self-hosted] unable to delete etcd data dir %s [%v]", cfg.Etcd.DataDir, err)
	}

	// Wait for the mirror Pod hash to be removed; otherwise we'll run into race conditions here when the kubelet hasn't had time to
	// remove the Static Pod (or the mirror Pod respectively). This implicitely also tests that the API server endpoint is healthy,
	// because this blocks until the API server returns a 404 Not Found when getting the Static Pod
	staticPodName := fmt.Sprintf("%s-%s", componentName, cfg.NodeName)
	if err := waiter.WaitForPodToDisappear(staticPodName); err != nil {
		return err
	}

	// Just as an extra safety check; make sure the API server is returning ok at the /healthz endpoint (although we know it could return a GET answer for a Pod above)
	if err := waiter.WaitForAPI(); err != nil {
		return err
	}

	fmt.Printf("[self-hosted] bootstrap %s deleted successfully after %f seconds\n", componentName, time.Since(start).Seconds())

	return nil
}
