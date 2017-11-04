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

package upgrade

import (
	"fmt"
	"io"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	versionutil "k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/pkg/version"
)

// VersionGetter defines an interface for fetching different versions.
// Easy to implement a fake variant of this interface for unit testing
type VersionGetter interface {
	// ClusterVersion should return the version of the cluster i.e. the API Server version
	ClusterVersion() (string, *versionutil.Version, error)
	// KubeadmVersion should return the version of the kubeadm CLI
	KubeadmVersion() (string, *versionutil.Version, error)
	// VersionFromCILabel should resolve CI labels like `latest`, `stable`, `stable-1.8`, etc. to real versions
	VersionFromCILabel(string, string) (string, *versionutil.Version, error)
	// KubeletVersions should return a map with a version and a number that describes how many kubelets there are for that version
	KubeletVersions() (map[string]uint16, error)
}

// KubeVersionGetter handles the version-fetching mechanism from external sources
type KubeVersionGetter struct {
	client clientset.Interface
	w      io.Writer
}

// NewKubeVersionGetter returns a new instance of KubeVersionGetter
func NewKubeVersionGetter(client clientset.Interface, writer io.Writer) VersionGetter {
	return &KubeVersionGetter{
		client: client,
		w:      writer,
	}
}

// ClusterVersion gets API server version
func (g *KubeVersionGetter) ClusterVersion() (string, *versionutil.Version, error) {
	clusterVersionInfo, err := g.client.Discovery().ServerVersion()
	if err != nil {
		return "", nil, fmt.Errorf("Couldn't fetch cluster version from the API Server: %v", err)
	}
	fmt.Fprintf(g.w, "[upgrade/versions] Cluster version: %s\n", clusterVersionInfo.String())

	clusterVersion, err := versionutil.ParseSemantic(clusterVersionInfo.String())
	if err != nil {
		return "", nil, fmt.Errorf("Couldn't parse cluster version: %v", err)
	}
	return clusterVersionInfo.String(), clusterVersion, nil
}

// KubeadmVersion gets kubeadm version
func (g *KubeVersionGetter) KubeadmVersion() (string, *versionutil.Version, error) {
	kubeadmVersionInfo := version.Get()
	fmt.Fprintf(g.w, "[upgrade/versions] kubeadm version: %s\n", kubeadmVersionInfo.String())

	kubeadmVersion, err := versionutil.ParseSemantic(kubeadmVersionInfo.String())
	if err != nil {
		return "", nil, fmt.Errorf("Couldn't parse kubeadm version: %v", err)
	}
	return kubeadmVersionInfo.String(), kubeadmVersion, nil
}

// VersionFromCILabel resolves a version label like "latest" or "stable" to an actual version using the public Kubernetes CI uploads
func (g *KubeVersionGetter) VersionFromCILabel(ciVersionLabel, description string) (string, *versionutil.Version, error) {
	versionStr, err := kubeadmutil.KubernetesReleaseVersion(ciVersionLabel)
	if err != nil {
		return "", nil, fmt.Errorf("Couldn't fetch latest %s from the internet: %v", description, err)
	}

	if description != "" {
		fmt.Fprintf(g.w, "[upgrade/versions] Latest %s: %s\n", description, versionStr)
	}

	ver, err := versionutil.ParseSemantic(versionStr)
	if err != nil {
		return "", nil, fmt.Errorf("Couldn't parse latest %s: %v", description, err)
	}
	return versionStr, ver, nil
}

// KubeletVersions gets the versions of the kubelets in the cluster
func (g *KubeVersionGetter) KubeletVersions() (map[string]uint16, error) {
	nodes, err := g.client.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("couldn't list all nodes in cluster")
	}
	return computeKubeletVersions(nodes.Items), nil
}

// computeKubeletVersions returns a string-int map that describes how many nodes are of a specific version
func computeKubeletVersions(nodes []v1.Node) map[string]uint16 {
	kubeletVersions := map[string]uint16{}
	for _, node := range nodes {
		kver := node.Status.NodeInfo.KubeletVersion
		if _, found := kubeletVersions[kver]; !found {
			kubeletVersions[kver] = 1
			continue
		}
		kubeletVersions[kver]++
	}
	return kubeletVersions
}
