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

package node

import (
	"fmt"
	"runtime"
	"runtime/pprof"
	"testing"

	"os"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

func TestAuthorizer(t *testing.T) {
	g := NewGraph()

	opts := sampleDataOpts{
		nodes:                  2,
		namespaces:             2,
		podsPerNode:            2,
		sharedConfigMapsPerPod: 0,
		uniqueConfigMapsPerPod: 1,
		sharedSecretsPerPod:    1,
		uniqueSecretsPerPod:    1,
		sharedPVCsPerPod:       0,
		uniquePVCsPerPod:       1,
	}
	pods, pvs := generate(opts)
	populate(g, pods, pvs)

	identifier := nodeidentifier.NewDefaultNodeIdentifier()
	authz := NewAuthorizer(g, identifier, bootstrappolicy.NodeRules())

	node0 := &user.DefaultInfo{Name: "system:node:node0", Groups: []string{"system:nodes"}}

	tests := []struct {
		name   string
		attrs  authorizer.AttributesRecord
		expect bool
	}{
		{
			name:   "allowed configmap",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "configmaps", Name: "configmap0-pod0-node0", Namespace: "ns0"},
			expect: true,
		},
		{
			name:   "allowed secret via pod",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "secrets", Name: "secret0-pod0-node0", Namespace: "ns0"},
			expect: true,
		},
		{
			name:   "allowed shared secret via pod",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "secrets", Name: "secret0-shared", Namespace: "ns0"},
			expect: true,
		},
		{
			name:   "allowed shared secret via pvc",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "secrets", Name: "secret-pv0-pod0-node0-ns0", Namespace: "ns0"},
			expect: true,
		},
		{
			name:   "allowed pvc",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "persistentvolumeclaims", Name: "pvc0-pod0-node0", Namespace: "ns0"},
			expect: true,
		},
		{
			name:   "allowed pv",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "persistentvolumes", Name: "pv0-pod0-node0-ns0", Namespace: ""},
			expect: true,
		},

		{
			name:   "disallowed configmap",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "configmaps", Name: "configmap0-pod0-node1", Namespace: "ns0"},
			expect: false,
		},
		{
			name:   "disallowed secret via pod",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "secrets", Name: "secret0-pod0-node1", Namespace: "ns0"},
			expect: false,
		},
		{
			name:   "disallowed shared secret via pvc",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "secrets", Name: "secret-pv0-pod0-node1-ns0", Namespace: "ns0"},
			expect: false,
		},
		{
			name:   "disallowed pvc",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "persistentvolumeclaims", Name: "pvc0-pod0-node1", Namespace: "ns0"},
			expect: false,
		},
		{
			name:   "disallowed pv",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "persistentvolumes", Name: "pv0-pod0-node1-ns0", Namespace: ""},
			expect: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ok, _, _ := authz.Authorize(tc.attrs)
			if ok != tc.expect {
				t.Errorf("expected %v, got %v", tc.expect, ok)
			}
		})
	}
}

func TestAuthorizerSharedResources(t *testing.T) {
	g := NewGraph()
	identifier := nodeidentifier.NewDefaultNodeIdentifier()
	authz := NewAuthorizer(g, identifier, bootstrappolicy.NodeRules())

	node1 := &user.DefaultInfo{Name: "system:node:node1", Groups: []string{"system:nodes"}}
	node2 := &user.DefaultInfo{Name: "system:node:node2", Groups: []string{"system:nodes"}}
	node3 := &user.DefaultInfo{Name: "system:node:node3", Groups: []string{"system:nodes"}}

	g.AddPod(&api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1-node1", Namespace: "ns1"},
		Spec: api.PodSpec{
			NodeName: "node1",
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "node1-only"}}},
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "node1-node2-only"}}},
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "shared-all"}}},
			},
		},
	})
	g.AddPod(&api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod2-node2", Namespace: "ns1"},
		Spec: api.PodSpec{
			NodeName: "node2",
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "node1-node2-only"}}},
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "shared-all"}}},
			},
		},
	})
	g.AddPod(&api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod3-node3", Namespace: "ns1"},
		Spec: api.PodSpec{
			NodeName: "node3",
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "shared-all"}}},
			},
		},
	})

	testcases := []struct {
		User          user.Info
		Secret        string
		ExpectAllowed bool
	}{
		{User: node1, ExpectAllowed: true, Secret: "node1-only"},
		{User: node1, ExpectAllowed: true, Secret: "node1-node2-only"},
		{User: node1, ExpectAllowed: true, Secret: "shared-all"},

		{User: node2, ExpectAllowed: false, Secret: "node1-only"},
		{User: node2, ExpectAllowed: true, Secret: "node1-node2-only"},
		{User: node2, ExpectAllowed: true, Secret: "shared-all"},

		{User: node3, ExpectAllowed: false, Secret: "node1-only"},
		{User: node3, ExpectAllowed: false, Secret: "node1-node2-only"},
		{User: node3, ExpectAllowed: true, Secret: "shared-all"},
	}

	for i, tc := range testcases {
		ok, _, err := authz.Authorize(authorizer.AttributesRecord{User: tc.User, ResourceRequest: true, Verb: "get", Resource: "secrets", Namespace: "ns1", Name: tc.Secret})
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		if ok != tc.ExpectAllowed {
			t.Errorf("%d: expected %v, got %v", i, tc.ExpectAllowed, ok)
		}
	}
}

type sampleDataOpts struct {
	nodes int

	namespaces int

	podsPerNode int

	sharedConfigMapsPerPod int
	sharedSecretsPerPod    int
	sharedPVCsPerPod       int

	uniqueSecretsPerPod    int
	uniqueConfigMapsPerPod int
	uniquePVCsPerPod       int
}

func BenchmarkPopulationAllocation(b *testing.B) {
	opts := sampleDataOpts{
		nodes:                  500,
		namespaces:             200,
		podsPerNode:            200,
		sharedConfigMapsPerPod: 0,
		uniqueConfigMapsPerPod: 1,
		sharedSecretsPerPod:    1,
		uniqueSecretsPerPod:    1,
		sharedPVCsPerPod:       0,
		uniquePVCsPerPod:       1,
	}

	pods, pvs := generate(opts)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		g := NewGraph()
		populate(g, pods, pvs)
	}
}

func BenchmarkPopulationRetention(b *testing.B) {

	// Run with:
	// go test ./plugin/pkg/auth/authorizer/node -benchmem -bench . -run None -v -o node.test -timeout 300m

	// Evaluate retained memory with:
	// go tool pprof --inuse_space node.test plugin/pkg/auth/authorizer/node/BenchmarkPopulationRetention.profile
	// list populate

	opts := sampleDataOpts{
		nodes:                  500,
		namespaces:             200,
		podsPerNode:            200,
		sharedConfigMapsPerPod: 0,
		uniqueConfigMapsPerPod: 1,
		sharedSecretsPerPod:    1,
		uniqueSecretsPerPod:    1,
		sharedPVCsPerPod:       0,
		uniquePVCsPerPod:       1,
	}

	pods, pvs := generate(opts)
	// Garbage collect before the first iteration
	runtime.GC()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		g := NewGraph()
		populate(g, pods, pvs)

		if i == 0 {
			f, _ := os.Create("BenchmarkPopulationRetention.profile")
			runtime.GC()
			pprof.WriteHeapProfile(f)
			f.Close()
			// reference the graph to keep it from getting garbage collected
			_ = fmt.Sprintf("%T\n", g)
		}
	}
}

func BenchmarkAuthorization(b *testing.B) {
	g := NewGraph()

	opts := sampleDataOpts{
		nodes:                  500,
		namespaces:             200,
		podsPerNode:            200,
		sharedConfigMapsPerPod: 0,
		uniqueConfigMapsPerPod: 1,
		sharedSecretsPerPod:    1,
		uniqueSecretsPerPod:    1,
		sharedPVCsPerPod:       0,
		uniquePVCsPerPod:       1,
	}
	pods, pvs := generate(opts)
	populate(g, pods, pvs)

	identifier := nodeidentifier.NewDefaultNodeIdentifier()
	authz := NewAuthorizer(g, identifier, bootstrappolicy.NodeRules())

	node0 := &user.DefaultInfo{Name: "system:node:node0", Groups: []string{"system:nodes"}}

	tests := []struct {
		name   string
		attrs  authorizer.AttributesRecord
		expect bool
	}{
		{
			name:   "allowed configmap",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "configmaps", Name: "configmap0-pod0-node0", Namespace: "ns0"},
			expect: true,
		},
		{
			name:   "allowed secret via pod",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "secrets", Name: "secret0-pod0-node0", Namespace: "ns0"},
			expect: true,
		},
		{
			name:   "allowed shared secret via pod",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "secrets", Name: "secret0-shared", Namespace: "ns0"},
			expect: true,
		},
		{
			name:   "disallowed configmap",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "configmaps", Name: "configmap0-pod0-node1", Namespace: "ns0"},
			expect: false,
		},
		{
			name:   "disallowed secret via pod",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "secrets", Name: "secret0-pod0-node1", Namespace: "ns0"},
			expect: false,
		},
		{
			name:   "disallowed shared secret via pvc",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "secrets", Name: "secret-pv0-pod0-node1-ns0", Namespace: "ns0"},
			expect: false,
		},
		{
			name:   "disallowed pvc",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "persistentvolumeclaims", Name: "pvc0-pod0-node1", Namespace: "ns0"},
			expect: false,
		},
		{
			name:   "disallowed pv",
			attrs:  authorizer.AttributesRecord{User: node0, ResourceRequest: true, Verb: "get", Resource: "persistentvolumes", Name: "pv0-pod0-node1-ns0", Namespace: ""},
			expect: false,
		},
	}

	b.ResetTimer()
	for _, tc := range tests {
		b.Run(tc.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ok, _, _ := authz.Authorize(tc.attrs)
				if ok != tc.expect {
					b.Errorf("expected %v, got %v", tc.expect, ok)
				}
			}
		})
	}
}

func populate(graph *Graph, pods []*api.Pod, pvs []*api.PersistentVolume) {
	p := &graphPopulator{}
	p.graph = graph
	for _, pod := range pods {
		p.addPod(pod)
	}
	for _, pv := range pvs {
		p.addPV(pv)
	}
}

// generate creates sample pods and persistent volumes based on the provided options.
// the secret/configmap/pvc/node references in the pod and pv objects are named to indicate the connections between the objects.
// for example, secret0-pod0-node0 is a secret referenced by pod0 which is bound to node0.
// when populated into the graph, the node authorizer should allow node0 to access that secret, but not node1.
func generate(opts sampleDataOpts) ([]*api.Pod, []*api.PersistentVolume) {
	pods := make([]*api.Pod, 0, opts.nodes*opts.podsPerNode)
	pvs := make([]*api.PersistentVolume, 0, (opts.nodes*opts.podsPerNode*opts.uniquePVCsPerPod)+(opts.sharedPVCsPerPod*opts.namespaces))

	for n := 0; n < opts.nodes; n++ {
		nodeName := fmt.Sprintf("node%d", n)
		for p := 0; p < opts.podsPerNode; p++ {
			pod := &api.Pod{}
			pod.Namespace = fmt.Sprintf("ns%d", p%opts.namespaces)
			pod.Name = fmt.Sprintf("pod%d-%s", p, nodeName)
			pod.Spec.NodeName = nodeName

			for i := 0; i < opts.uniqueSecretsPerPod; i++ {
				pod.Spec.Volumes = append(pod.Spec.Volumes, api.Volume{VolumeSource: api.VolumeSource{
					Secret: &api.SecretVolumeSource{SecretName: fmt.Sprintf("secret%d-%s", i, pod.Name)},
				}})
			}
			for i := 0; i < opts.sharedSecretsPerPod; i++ {
				pod.Spec.Volumes = append(pod.Spec.Volumes, api.Volume{VolumeSource: api.VolumeSource{
					Secret: &api.SecretVolumeSource{SecretName: fmt.Sprintf("secret%d-shared", i)},
				}})
			}

			for i := 0; i < opts.uniqueConfigMapsPerPod; i++ {
				pod.Spec.Volumes = append(pod.Spec.Volumes, api.Volume{VolumeSource: api.VolumeSource{
					ConfigMap: &api.ConfigMapVolumeSource{LocalObjectReference: api.LocalObjectReference{Name: fmt.Sprintf("configmap%d-%s", i, pod.Name)}},
				}})
			}
			for i := 0; i < opts.sharedConfigMapsPerPod; i++ {
				pod.Spec.Volumes = append(pod.Spec.Volumes, api.Volume{VolumeSource: api.VolumeSource{
					ConfigMap: &api.ConfigMapVolumeSource{LocalObjectReference: api.LocalObjectReference{Name: fmt.Sprintf("configmap%d-shared", i)}},
				}})
			}

			for i := 0; i < opts.uniquePVCsPerPod; i++ {
				pv := &api.PersistentVolume{}
				pv.Name = fmt.Sprintf("pv%d-%s-%s", i, pod.Name, pod.Namespace)
				pv.Spec.FlexVolume = &api.FlexVolumeSource{SecretRef: &api.LocalObjectReference{Name: fmt.Sprintf("secret-%s", pv.Name)}}
				pv.Spec.ClaimRef = &api.ObjectReference{Name: fmt.Sprintf("pvc%d-%s", i, pod.Name), Namespace: pod.Namespace}
				pvs = append(pvs, pv)

				pod.Spec.Volumes = append(pod.Spec.Volumes, api.Volume{VolumeSource: api.VolumeSource{
					PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{ClaimName: pv.Spec.ClaimRef.Name},
				}})
			}
			for i := 0; i < opts.sharedPVCsPerPod; i++ {
				pv := &api.PersistentVolume{}
				pv.Name = fmt.Sprintf("pv%d-shared-%s", i, pod.Namespace)
				pv.Spec.FlexVolume = &api.FlexVolumeSource{SecretRef: &api.LocalObjectReference{Name: fmt.Sprintf("secret-%s", pv.Name)}}
				pv.Spec.ClaimRef = &api.ObjectReference{Name: fmt.Sprintf("pvc%d-shared", i), Namespace: pod.Namespace}
				pvs = append(pvs, pv)

				pod.Spec.Volumes = append(pod.Spec.Volumes, api.Volume{VolumeSource: api.VolumeSource{
					PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{ClaimName: pv.Spec.ClaimRef.Name},
				}})
			}

			pods = append(pods, pod)
		}
	}
	return pods, pvs
}
