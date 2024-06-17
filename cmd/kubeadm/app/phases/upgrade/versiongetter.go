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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	versionutil "k8s.io/apimachinery/pkg/util/version"
	pkgversion "k8s.io/apimachinery/pkg/version"
	fakediscovery "k8s.io/client-go/discovery/fake"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/version"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/image"
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
	// KubeletVersions should return a map with a version and a list of node names that describes how many kubelets there are for that version
	KubeletVersions() (map[string][]string, error)
	// ComponentVersions should return a map with a version and a list of node names that describes how many a given control-plane components there are for that version
	ComponentVersions(string) (map[string][]string, error)
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
	var (
		clusterVersionInfo *pkgversion.Info
		err                error
	)
	// If we are dry-running, do not attempt to fetch the /version resource and just return
	// the stored FakeServerVersion, which is done when constructing the dry-run client in
	// common.go#getClient()
	// The problem here is that during upgrade dry-run client reactors are backed by a dynamic client
	// via NewClientBackedDryRunGetterFromKubeconfig() and for GetActions there seems to be no analog to
	// Discovery().Serverversion() resource for a dynamic client(?).
	fakeclientDiscovery, ok := g.client.Discovery().(*fakediscovery.FakeDiscovery)
	if ok {
		clusterVersionInfo = fakeclientDiscovery.FakedServerVersion
	} else {
		clusterVersionInfo, err = g.client.Discovery().ServerVersion()
		if err != nil {
			return "", nil, errors.Wrap(err, "Couldn't fetch cluster version from the API Server")
		}
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

// KubeletVersions gets the versions of the kubelets in the cluster.
func (g *KubeVersionGetter) KubeletVersions() (map[string][]string, error) {
	nodes, err := g.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, errors.New("couldn't list all nodes in cluster")
	}

	// map kubelet version to a list of node names
	kubeletVersions := make(map[string][]string)
	for _, node := range nodes.Items {
		kver := node.Status.NodeInfo.KubeletVersion
		kubeletVersions[kver] = append(kubeletVersions[kver], node.Name)
	}
	return kubeletVersions, nil
}

// ComponentVersions gets the versions of the control-plane components in the cluster.
// The name parameter is the name of the component to get the versions for.
// The function returns a map with the version as the key and a list of node names as the value.
func (g *KubeVersionGetter) ComponentVersions(name string) (map[string][]string, error) {
	podList, err := g.client.CoreV1().Pods(metav1.NamespaceSystem).List(
		context.TODO(),
		metav1.ListOptions{
			LabelSelector: fmt.Sprintf("component=%s,tier=%s", name, constants.ControlPlaneTier),
		},
	)
	if err != nil {
		return nil, errors.Wrap(err, "couldn't list pods in cluster")
	}

	componentVersions := make(map[string][]string)
	for _, pod := range podList.Items {
		tag := convertImageTagMetadataToSemver(image.TagFromImage(pod.Spec.Containers[0].Image))
		componentVersions[tag] = append(componentVersions[tag], pod.Spec.NodeName)
	}
	return componentVersions, nil
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
		return o.VersionGetter.VersionFromCILabel(ciVersionLabel, description)
	}
	ver, err := versionutil.ParseSemantic(o.version)
	if err != nil {
		return "", nil, errors.Wrapf(err, "Couldn't parse version %s", description)
	}
	return o.version, ver, nil
}
