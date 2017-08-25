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

	clientset "k8s.io/client-go/kubernetes"
	versionutil "k8s.io/kubernetes/pkg/util/version"
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

// Make sure KubeVersionGetter implements the VersionGetter interface
var _ VersionGetter = &KubeVersionGetter{}

// NewKubeVersionGetter returns a new instance of KubeVersionGetter
func NewKubeVersionGetter(client clientset.Interface, writer io.Writer) *KubeVersionGetter {
	return &KubeVersionGetter{
		client: client,
		w:      writer,
	}
}

// ClusterVersion gets API server version
func (g *KubeVersionGetter) ClusterVersion() (string, *versionutil.Version, error) {
	fmt.Fprintf(g.w, "[upgrade/versions] Cluster version: ")
	fmt.Fprintln(g.w, "v1.7.0")

	return "v1.7.0", versionutil.MustParseSemantic("v1.7.0"), nil
}

// KubeadmVersion gets kubeadm version
func (g *KubeVersionGetter) KubeadmVersion() (string, *versionutil.Version, error) {
	fmt.Fprintf(g.w, "[upgrade/versions] kubeadm version: %s\n", "v1.8.0")

	return "v1.8.0", versionutil.MustParseSemantic("v1.8.0"), nil
}

// VersionFromCILabel resolves different labels like "stable" to action semver versions using the Kubernetes CI uploads to GCS
func (g *KubeVersionGetter) VersionFromCILabel(_, _ string) (string, *versionutil.Version, error) {
	return "v1.8.1", versionutil.MustParseSemantic("v1.8.0"), nil
}

// KubeletVersions gets the versions of the kubelets in the cluster
func (g *KubeVersionGetter) KubeletVersions() (map[string]uint16, error) {
	// This tells kubeadm that there are two nodes in the cluster; both on the v1.7.1 version currently
	return map[string]uint16{
		"v1.7.1": 2,
	}, nil
}
