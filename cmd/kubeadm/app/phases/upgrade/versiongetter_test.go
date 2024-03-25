/*
Copyright 2023 The Kubernetes Authors.

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
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	fakediscovery "k8s.io/client-go/discovery/fake"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
)

func TestKubeVersionGetterClusterVersion(t *testing.T) {
	tests := []struct {
		name               string
		version            *version.Info
		wantClusterVersion string
		wantParsedVersion  string
		wantErr            bool
	}{
		{
			name: "cluster version is valid",
			version: &version.Info{
				GitVersion: "1.20.0",
			},
			wantClusterVersion: "1.20.0",
			wantParsedVersion:  "1.20.0",
			wantErr:            false,
		},
		{
			name:               "cluster version is empty",
			version:            &version.Info{},
			wantClusterVersion: "",
			wantParsedVersion:  "",
			wantErr:            true,
		},
		{
			name: "cluster version is invalid",
			version: &version.Info{
				GitVersion: "invalid-version",
			},
			wantClusterVersion: "",
			wantParsedVersion:  "",
			wantErr:            true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			client.Discovery().(*fakediscovery.FakeDiscovery).FakedServerVersion = tt.version

			g := &KubeVersionGetter{
				client: client,
			}
			clusterVersion, parsedVersion, err := g.ClusterVersion()

			if (err != nil) != tt.wantErr {
				t.Errorf("error = %v, wantErr = %v", err, tt.wantErr)
				return
			}
			if clusterVersion != tt.wantClusterVersion {
				t.Errorf("clusterVersion = %v, wantClusterVersion = %v", clusterVersion, tt.wantClusterVersion)
			}
			if tt.wantParsedVersion == "" {
				if parsedVersion != nil {
					t.Errorf("parsedVersion = %v, wantParsedVersion = %v", parsedVersion, tt.wantParsedVersion)
				}
			} else if parsedVersion.String() != tt.wantParsedVersion {
				t.Errorf("parsedVersion = %v, wantParsedVersion = %v", parsedVersion, tt.wantParsedVersion)
			}
		})
	}
}

func TestKubeVersionGetterVersionFromCILabel(t *testing.T) {
	tests := []struct {
		name              string
		ciVersionLabel    string
		wantCIVersion     string
		wantParsedVersion string
		wantErr           bool
	}{
		{
			name:              "version is valid",
			ciVersionLabel:    "v1.28.1",
			wantCIVersion:     "v1.28.1",
			wantParsedVersion: "1.28.1",
			wantErr:           false,
		},
		{
			name:              "version is invalid",
			ciVersionLabel:    "invalid-version",
			wantCIVersion:     "",
			wantParsedVersion: "",
			wantErr:           true,
		},
		{
			name:              "version is empty",
			ciVersionLabel:    "",
			wantCIVersion:     "",
			wantParsedVersion: "",
			wantErr:           true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &KubeVersionGetter{
				client: clientsetfake.NewSimpleClientset(),
			}
			cliVersion, parsedVersion, err := g.VersionFromCILabel(tt.ciVersionLabel, "test VersionFromCILabel")
			if (err != nil) != tt.wantErr {
				t.Errorf("error = %v, wantErr = %v", err, tt.wantErr)
				return
			}
			if cliVersion != tt.wantCIVersion {
				t.Errorf("cliVersion = %v, wantCIVersion = %v", cliVersion, tt.wantCIVersion)
			}
			if tt.wantParsedVersion == "" {
				if parsedVersion != nil {
					t.Errorf("parsedVersion = %v, wantParsedVersion = %v", parsedVersion, tt.wantParsedVersion)
				}
			} else if parsedVersion.String() != tt.wantParsedVersion {
				t.Errorf("parsedVersion = %v, wantParsedVersion = %v", parsedVersion, tt.wantParsedVersion)
			}
		})
	}
}

func TestKubeVersionGetterKubeletVersions(t *testing.T) {
	tests := []struct {
		name    string
		nodes   *v1.NodeList
		want    map[string][]string
		wantErr bool
	}{
		{
			name: "kubelet version info exists",
			nodes: &v1.NodeList{
				Items: []v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "node1"},
						Status: v1.NodeStatus{
							NodeInfo: v1.NodeSystemInfo{
								KubeletVersion: "v1.28.0",
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "node2"},
						Status: v1.NodeStatus{
							NodeInfo: v1.NodeSystemInfo{
								KubeletVersion: "v1.28.1",
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "node3"},
						Status: v1.NodeStatus{
							NodeInfo: v1.NodeSystemInfo{
								KubeletVersion: "v1.28.1",
							},
						},
					},
				},
			},
			want: map[string][]string{
				"v1.28.0": {"node1"},
				"v1.28.1": {"node2", "node3"},
			},
			wantErr: false,
		},
		{
			name: "kubelet version info is empty",
			nodes: &v1.NodeList{
				Items: []v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "node2"},
						Status:     v1.NodeStatus{},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "node3"},
						Status:     v1.NodeStatus{},
					},
				},
			},
			want: map[string][]string{
				"": {"node2", "node3"},
			},
			wantErr: false,
		},
		{
			name:    "node list is empty",
			nodes:   &v1.NodeList{},
			want:    map[string][]string{},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		client := clientsetfake.NewSimpleClientset()
		t.Run(tt.name, func(t *testing.T) {
			for _, node := range tt.nodes.Items {
				err := client.Tracker().Create(schema.GroupVersionResource{Version: "v1", Resource: "nodes"}, &node, "")
				if err != nil {
					t.Fatal(err)
				}
			}
			g := &KubeVersionGetter{
				client: client,
			}
			got, err := g.KubeletVersions()
			if (err != nil) != tt.wantErr {
				t.Errorf("error = %v, wantErr = %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got = %v, want = %v", got, tt.want)
			}
		})
	}
}
