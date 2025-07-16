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
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	utilsnet "k8s.io/utils/net"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/users"
)

const (
	etcdVolumeName           = "etcd-data"
	certsVolumeName          = "etcd-certs"
	etcdHealthyCheckInterval = 5 * time.Second
	etcdHealthyCheckRetries  = 8
)

// CreateLocalEtcdStaticPodManifestFile will write local etcd static pod manifest file.
// This function is used by init - when the etcd cluster is empty - or by kubeadm
// upgrade - when the etcd cluster is already up and running (and the --initial-cluster flag have no impact)
func CreateLocalEtcdStaticPodManifestFile(manifestDir, patchesDir string, nodeName string, cfg *kubeadmapi.ClusterConfiguration, endpoint *kubeadmapi.APIEndpoint, isDryRun bool) error {
	if cfg.Etcd.External != nil {
		return errors.New("etcd static pod manifest cannot be generated for cluster using external etcd")
	}

	if err := prepareAndWriteEtcdStaticPod(manifestDir, patchesDir, cfg, endpoint, nodeName, []etcdutil.Member{}, isDryRun); err != nil {
		return err
	}

	klog.V(1).Infof("[etcd] wrote Static Pod manifest for a local etcd member to %q\n", kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.Etcd, manifestDir))
	return nil
}

// CheckLocalEtcdClusterStatus verifies health state of local/stacked etcd cluster before installing a new etcd member
func CheckLocalEtcdClusterStatus(client clientset.Interface, certificatesDir string) error {
	klog.V(1).Info("[etcd] Checking etcd cluster health")

	// creates an etcd client that connects to all the local/stacked etcd members
	klog.V(1).Info("creating etcd client that connects to etcd pods")
	etcdClient, err := etcdutil.NewFromCluster(client, certificatesDir)
	if err != nil {
		return err
	}

	// Checking health state
	err = etcdClient.CheckClusterHealth()
	if err != nil {
		return errors.Wrap(err, "etcd cluster is not healthy")
	}

	return nil
}

// RemoveStackedEtcdMemberFromCluster will remove a local etcd member from etcd cluster,
// when reset the control plane node.
func RemoveStackedEtcdMemberFromCluster(client clientset.Interface, cfg *kubeadmapi.InitConfiguration) error {
	// creates an etcd client that connects to all the local/stacked etcd members
	klog.V(1).Info("[etcd] creating etcd client that connects to etcd pods")
	etcdClient, err := etcdutil.NewFromCluster(client, cfg.CertificatesDir)
	if err != nil {
		return err
	}

	members, err := etcdClient.ListMembers()
	if err != nil {
		return err
	}
	// If this is the only remaining stacked etcd member in the cluster, calling RemoveMember()
	// is not needed.
	if len(members) == 1 {
		etcdClientAddress := etcdutil.GetClientURL(&cfg.LocalAPIEndpoint)
		for _, endpoint := range etcdClient.Endpoints {
			if endpoint == etcdClientAddress {
				klog.V(1).Info("[etcd] This is the only remaining etcd member in the etcd cluster, skip removing it")
				return nil
			}
		}
	}

	// notifies the other members of the etcd cluster about the removing member
	etcdPeerAddress := etcdutil.GetPeerURL(&cfg.LocalAPIEndpoint)

	klog.V(2).Infof("[etcd] get the member id from peer: %s", etcdPeerAddress)
	id, err := etcdClient.GetMemberID(etcdPeerAddress)
	if err != nil {
		if errors.Is(etcdutil.ErrNoMemberIDForPeerURL, err) {
			klog.V(5).Infof("[etcd] member was already removed, because no member id exists for peer %s", etcdPeerAddress)
			return nil
		}
		return err
	}

	klog.V(1).Infof("[etcd] removing etcd member: %s, id: %d", etcdPeerAddress, id)
	members, err = etcdClient.RemoveMember(id)
	if err != nil {
		return err
	}
	klog.V(1).Infof("[etcd] Updated etcd member list: %v", members)

	return nil
}

// CreateStackedEtcdStaticPodManifestFile will write local etcd static pod manifest file
// for an additional etcd member that is joining an existing local/stacked etcd cluster.
// Other members of the etcd cluster will be notified of the joining node in beforehand as well.
func CreateStackedEtcdStaticPodManifestFile(client clientset.Interface, manifestDir, patchesDir string, nodeName string, cfg *kubeadmapi.ClusterConfiguration, endpoint *kubeadmapi.APIEndpoint, isDryRun bool, certificatesDir string) error {
	etcdPeerAddress := etcdutil.GetPeerURL(endpoint)

	var cluster []etcdutil.Member
	var etcdClient *etcdutil.Client
	var err error
	if isDryRun {
		fmt.Printf("[etcd] Would add etcd member: %s\n", etcdPeerAddress)
	} else {
		// Creates an etcd client that connects to all the local/stacked etcd members.
		klog.V(1).Info("creating etcd client that connects to etcd pods")
		etcdClient, err = etcdutil.NewFromCluster(client, certificatesDir)
		if err != nil {
			return err
		}
		klog.V(1).Infof("[etcd] Adding etcd member: %s", etcdPeerAddress)
		cluster, err = etcdClient.AddMemberAsLearner(nodeName, etcdPeerAddress)
		if err != nil {
			return err
		}
		fmt.Println("[etcd] Announced new etcd member joining to the existing etcd cluster")
		klog.V(1).Infof("Updated etcd member list: %v", cluster)
	}

	fmt.Printf("[etcd] Creating static Pod manifest for %q\n", kubeadmconstants.Etcd)

	if err := prepareAndWriteEtcdStaticPod(manifestDir, patchesDir, cfg, endpoint, nodeName, cluster, isDryRun); err != nil {
		return err
	}

	if isDryRun {
		fmt.Println("[etcd] Would wait for the new etcd member to join the cluster")
		return nil
	}

	learnerID, err := etcdClient.GetMemberID(etcdPeerAddress)
	if err != nil {
		return err
	}
	err = etcdClient.MemberPromote(learnerID)
	if err != nil {
		return err
	}

	fmt.Printf("[etcd] Waiting for the new etcd member to join the cluster. This can take up to %v\n", etcdHealthyCheckInterval*etcdHealthyCheckRetries)
	if _, err := etcdClient.WaitForClusterAvailable(etcdHealthyCheckRetries, etcdHealthyCheckInterval); err != nil {
		return err
	}

	return nil
}

// GetEtcdPodSpec returns the etcd static Pod actualized to the context of the current configuration
// NB. GetEtcdPodSpec methods holds the information about how kubeadm creates etcd static pod manifests.
func GetEtcdPodSpec(cfg *kubeadmapi.ClusterConfiguration, endpoint *kubeadmapi.APIEndpoint, nodeName string, initialCluster []etcdutil.Member) v1.Pod {
	pathType := v1.HostPathDirectoryOrCreate
	etcdMounts := map[string]v1.Volume{
		etcdVolumeName:  staticpodutil.NewVolume(etcdVolumeName, cfg.Etcd.Local.DataDir, &pathType),
		certsVolumeName: staticpodutil.NewVolume(certsVolumeName, cfg.CertificatesDir+"/etcd", &pathType),
	}
	componentHealthCheckTimeout := kubeadmapi.GetActiveTimeouts().ControlPlaneComponentHealthCheck

	// probeHostname returns the correct localhost IP address family based on the endpoint AdvertiseAddress
	probeHostname, probePort, probeScheme := staticpodutil.GetEtcdProbeEndpoint(&cfg.Etcd, utilsnet.IsIPv6String(endpoint.AdvertiseAddress))
	return staticpodutil.ComponentPod(
		v1.Container{
			Name:            kubeadmconstants.Etcd,
			Command:         getEtcdCommand(cfg, endpoint, nodeName, initialCluster),
			Image:           images.GetEtcdImage(cfg),
			ImagePullPolicy: v1.PullIfNotPresent,
			// Mount the etcd datadir path read-write so etcd can store data in a more persistent manner
			VolumeMounts: []v1.VolumeMount{
				staticpodutil.NewVolumeMount(etcdVolumeName, cfg.Etcd.Local.DataDir, false),
				staticpodutil.NewVolumeMount(certsVolumeName, cfg.CertificatesDir+"/etcd", false),
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			},
			// The etcd probe endpoints are explained here:
			// https://github.com/kubernetes/kubeadm/issues/3039
			LivenessProbe:  staticpodutil.LivenessProbe(probeHostname, "/livez", probePort, probeScheme),
			ReadinessProbe: staticpodutil.ReadinessProbe(probeHostname, "/readyz", probePort, probeScheme),
			StartupProbe:   staticpodutil.StartupProbe(probeHostname, "/readyz", probePort, probeScheme, componentHealthCheckTimeout),
			Env:            kubeadmutil.MergeKubeadmEnvVars(cfg.Etcd.Local.ExtraEnvs),
		},
		etcdMounts,
		// etcd will listen on the advertise address of the API server, in a different port (2379)
		map[string]string{kubeadmconstants.EtcdAdvertiseClientUrlsAnnotationKey: etcdutil.GetClientURL(endpoint)},
	)
}

// getEtcdCommand builds the right etcd command from the given config object
func getEtcdCommand(cfg *kubeadmapi.ClusterConfiguration, endpoint *kubeadmapi.APIEndpoint, nodeName string, initialCluster []etcdutil.Member) []string {
	// localhost IP family should be the same that the AdvertiseAddress
	etcdLocalhostAddress := "127.0.0.1"
	if utilsnet.IsIPv6String(endpoint.AdvertiseAddress) {
		etcdLocalhostAddress = "::1"
	}
	defaultArguments := []kubeadmapi.Arg{
		{Name: "name", Value: nodeName},
		// TODO: start using --initial-corrupt-check once the graduated flag is available,
		// https://github.com/kubernetes/kubeadm/issues/2676
		{Name: "experimental-initial-corrupt-check", Value: "true"},
		{Name: "listen-client-urls", Value: fmt.Sprintf("%s,%s", etcdutil.GetClientURLByIP(etcdLocalhostAddress), etcdutil.GetClientURL(endpoint))},
		{Name: "advertise-client-urls", Value: etcdutil.GetClientURL(endpoint)},
		{Name: "listen-peer-urls", Value: etcdutil.GetPeerURL(endpoint)},
		{Name: "initial-advertise-peer-urls", Value: etcdutil.GetPeerURL(endpoint)},
		{Name: "data-dir", Value: cfg.Etcd.Local.DataDir},
		{Name: "cert-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdServerCertName)},
		{Name: "key-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdServerKeyName)},
		{Name: "trusted-ca-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdCACertName)},
		{Name: "client-cert-auth", Value: "true"},
		{Name: "peer-cert-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdPeerCertName)},
		{Name: "peer-key-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdPeerKeyName)},
		{Name: "peer-trusted-ca-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdCACertName)},
		{Name: "peer-client-cert-auth", Value: "true"},
		{Name: "snapshot-count", Value: "10000"},
		{Name: "listen-metrics-urls", Value: fmt.Sprintf("http://%s", net.JoinHostPort(etcdLocalhostAddress, strconv.Itoa(kubeadmconstants.EtcdMetricsPort)))},
		{Name: "experimental-watch-progress-notify-interval", Value: "5s"},
	}

	if len(initialCluster) == 0 {
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "initial-cluster", fmt.Sprintf("%s=%s", nodeName, etcdutil.GetPeerURL(endpoint)), 1)
	} else {
		// NB. the joining etcd member should be part of the initialCluster list
		endpoints := []string{}
		for _, member := range initialCluster {
			endpoints = append(endpoints, fmt.Sprintf("%s=%s", member.Name, member.PeerURL))
		}

		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "initial-cluster", strings.Join(endpoints, ","), 1)
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "initial-cluster-state", "existing", 1)
	}

	command := []string{"etcd"}
	command = append(command, kubeadmutil.ArgumentsToCommand(defaultArguments, cfg.Etcd.Local.ExtraArgs)...)
	return command
}

func prepareAndWriteEtcdStaticPod(manifestDir string, patchesDir string, cfg *kubeadmapi.ClusterConfiguration, endpoint *kubeadmapi.APIEndpoint, nodeName string, initialCluster []etcdutil.Member, isDryRun bool) error {
	// gets etcd StaticPodSpec, actualized for the current ClusterConfiguration and the new list of etcd members
	spec := GetEtcdPodSpec(cfg, endpoint, nodeName, initialCluster)

	var usersAndGroups *users.UsersAndGroups
	var err error
	if features.Enabled(cfg.FeatureGates, features.RootlessControlPlane) {
		if isDryRun {
			fmt.Printf("[etcd] Would create users and groups for %q to run as non-root\n", kubeadmconstants.Etcd)
			fmt.Printf("[etcd] Would update static pod manifest for %q to run run as non-root\n", kubeadmconstants.Etcd)
		} else {
			usersAndGroups, err = staticpodutil.GetUsersAndGroups()
			if err != nil {
				return errors.Wrap(err, "failed to create users and groups")
			}
			// usersAndGroups is nil on non-linux.
			if usersAndGroups != nil {
				if err := staticpodutil.RunComponentAsNonRoot(kubeadmconstants.Etcd, &spec, usersAndGroups, cfg); err != nil {
					return errors.Wrapf(err, "failed to run component %q as non-root", kubeadmconstants.Etcd)
				}
			}
		}
	}

	// if patchesDir is defined, patch the static Pod manifest
	if patchesDir != "" {
		patchedSpec, err := staticpodutil.PatchStaticPod(&spec, patchesDir, os.Stdout)
		if err != nil {
			return errors.Wrapf(err, "failed to patch static Pod manifest file for %q", kubeadmconstants.Etcd)
		}
		spec = *patchedSpec
	}

	// writes etcd StaticPod to disk
	if err := staticpodutil.WriteStaticPodToDisk(kubeadmconstants.Etcd, manifestDir, spec); err != nil {
		return err
	}

	// If dry-running, print the static etcd pod manifest file.
	if isDryRun {
		realPath := kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.Etcd, manifestDir)
		outputPath := kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.Etcd, kubeadmconstants.GetStaticPodDirectory())
		return dryrunutil.PrintDryRunFiles([]dryrunutil.FileToPrint{dryrunutil.NewFileToPrint(realPath, outputPath)}, os.Stdout)
	}
	return nil
}
