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
	"context"
	"fmt"
	"github.com/pkg/errors"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	versionutil "k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/version"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
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
}

// NewKubeVersionGetter returns a new instance of KubeVersionGetter
func NewKubeVersionGetter(client clientset.Interface) VersionGetter {
	return &KubeVersionGetter{
		client: client,
	}
}

// ClusterVersion gets API server version
func (g *KubeVersionGetter) ClusterVersion() (string, *versionutil.Version, error) {
	clusterVersionInfo, err := g.client.Discovery().ServerVersion()
	if err != nil {
		return "", nil, errors.Wrap(err, "Couldn't fetch cluster version from the API Server")
	}

	clusterVersion, err := versionutil.ParseSemantic(clusterVersionInfo.String())
	if err != nil {
		return "", nil, errors.Wrap(err, "Couldn't parse cluster version")
	}
	return clusterVersionInfo.String(), clusterVersion, nil
}

// KubeadmVersion gets kubeadm version
func (g *KubeVersionGetter) KubeadmVersion() (string, *versionutil.Version, error) {
	kubeadmVersionInfo := version.Get()

	kubeadmVersion, err := versionutil.ParseSemantic(kubeadmVersionInfo.String())
	if err != nil {
		return "", nil, errors.Wrap(err, "Couldn't parse kubeadm version")
	}
	return kubeadmVersionInfo.String(), kubeadmVersion, nil
}

// VersionFromCILabel resolves a version label like "latest" or "stable" to an actual version using the public Kubernetes CI uploads
func (g *KubeVersionGetter) VersionFromCILabel(ciVersionLabel, description string) (string, *versionutil.Version, error) {
	versionStr, err := kubeadmutil.KubernetesReleaseVersion(ciVersionLabel)
	if err != nil {
		return "", nil, errors.Wrapf(err, "Couldn't fetch latest %s from the internet", description)
	}

	ver, err := versionutil.ParseSemantic(versionStr)
	if err != nil {
		return "", nil, errors.Wrapf(err, "Couldn't parse latest %s", description)
	}
	return versionStr, ver, nil
}

// KubeletVersions gets the versions of the kubelets in the cluster
func (g *KubeVersionGetter) KubeletVersions() (map[string]uint16, error) {
	nodes, err := g.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, errors.New("couldn't list all nodes in cluster")
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

// OfflineVersionGetter will use the version provided or
type OfflineVersionGetter struct {
	VersionGetter
	version string
}

// NewOfflineVersionGetter wraps a VersionGetter and skips online communication if default information is supplied.
// Version can be "" and the behavior will be identical to the versionGetter passed in.
func NewOfflineVersionGetter(versionGetter VersionGetter, version string) VersionGetter {
	return &OfflineVersionGetter{
		VersionGetter: versionGetter,
		version:       version,
	}
}

// VersionFromCILabel will return the version that was passed into the struct
func (o *OfflineVersionGetter) VersionFromCILabel(ciVersionLabel, description string) (string, *versionutil.Version, error) {
	if o.version == "" {
		versionStr, version, err := o.VersionGetter.VersionFromCILabel(ciVersionLabel, description)
		if err == nil {
			fmt.Printf("[upgrade/versions] Latest %s: %s\n", description, versionStr)
		}
		return versionStr, version, err
	}
	ver, err := versionutil.ParseSemantic(o.version)
	if err != nil {
		return "", nil, errors.Wrapf(err, "Couldn't parse version %s", description)
	}
	return o.version, ver, nil
}
