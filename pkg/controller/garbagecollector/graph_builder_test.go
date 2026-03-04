/*
Copyright 2020 The Kubernetes Authors.

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

package garbagecollector

import (
	"reflect"
	"testing"
)

func TestGetAlternateOwnerIdentity(t *testing.T) {
	ns1child1 := makeID("v1", "Child", "ns1", "child1", "childuid11")
	ns1child2 := makeID("v1", "Child", "ns1", "child2", "childuid12")

	ns2child1 := makeID("v1", "Child", "ns2", "child1", "childuid21")

	clusterchild1 := makeID("v1", "Child", "", "child1", "childuidc1")

	var (
		nsabsentparentns1 = makeID("v1", "NSParent", "ns1", "parentname", "parentuid")
		nsabsentparentns2 = makeID("v1", "NSParent", "ns2", "parentname", "parentuid")

		nsabsentparent_version = makeID("xx", "NSParent", "ns1", "parentname", "parentuid")
		nsabsentparent_kind    = makeID("v1", "xxxxxxxx", "ns1", "parentname", "parentuid")
		nsabsentparent_name    = makeID("v1", "NSParent", "ns1", "xxxxxxxxxx", "parentuid")

		clusterabsentparent         = makeID("v1", "ClusterParent", "", "parentname", "parentuid")
		clusterabsentparent_version = makeID("xx", "ClusterParent", "", "parentname", "parentuid")
		clusterabsentparent_kind    = makeID("v1", "xxxxxxxxxxxxx", "", "parentname", "parentuid")
		clusterabsentparent_name    = makeID("v1", "ClusterParent", "", "xxxxxxxxxx", "parentuid")

		clusterabsentparent_ns1_version = makeID("xx", "ClusterParent", "ns1", "parentname", "parentuid")
		clusterabsentparent_ns1_kind    = makeID("v1", "xxxxxxxxxxxxx", "ns1", "parentname", "parentuid")
	)

	orderedNamespacedReferences := []objectReference{
		makeID("v1", "kind", "ns1", "name", "uid"),
		makeID("v2", "kind", "ns1", "name", "uid"),
		makeID("v3", "kind", "ns1", "name", "uid"),
		makeID("v4", "kind", "ns1", "name", "uid"),
		makeID("v5", "kind", "ns1", "name", "uid"),
	}
	orderedClusterReferences := []objectReference{
		makeID("v1", "kind", "", "name", "uid"),
		makeID("v2", "kind", "", "name", "uid"),
		makeID("v3", "kind", "", "name", "uid"),
		makeID("v4", "kind", "", "name", "uid"),
		makeID("v5", "kind", "", "name", "uid"),
	}

	testcases := []struct {
		name              string
		deps              []*node
		verifiedAbsent    objectReference
		expectedAlternate *objectReference
	}{
		{
			name: "namespaced alternate version",
			deps: []*node{
				makeNode(ns1child1, withOwners(nsabsentparentns1)),
				makeNode(ns1child2, withOwners(nsabsentparent_version)),
			},
			verifiedAbsent:    nsabsentparentns1,
			expectedAlternate: &nsabsentparent_version, // switch to alternate version
		},
		{
			name: "namespaced alternate kind",
			deps: []*node{
				makeNode(ns1child1, withOwners(nsabsentparentns1)),
				makeNode(ns1child2, withOwners(nsabsentparent_kind)),
			},
			verifiedAbsent:    nsabsentparentns1,
			expectedAlternate: &nsabsentparent_kind, // switch to alternate kind
		},
		{
			name: "namespaced alternate namespace",
			deps: []*node{
				makeNode(ns1child1, withOwners(nsabsentparentns1)),
				makeNode(ns2child1, withOwners(nsabsentparentns2)),
			},
			verifiedAbsent:    nsabsentparentns1,
			expectedAlternate: &nsabsentparentns2, // switch to alternate namespace
		},
		{
			name: "namespaced alternate name",
			deps: []*node{
				makeNode(ns1child1, withOwners(nsabsentparentns1)),
				makeNode(ns1child1, withOwners(nsabsentparent_name)),
			},
			verifiedAbsent:    nsabsentparentns1,
			expectedAlternate: &nsabsentparent_name, // switch to alternate name
		},

		{
			name: "cluster alternate version",
			deps: []*node{
				makeNode(ns1child1, withOwners(clusterabsentparent)),
				makeNode(ns1child2, withOwners(clusterabsentparent_version)),
			},
			verifiedAbsent:    clusterabsentparent,
			expectedAlternate: &clusterabsentparent_ns1_version, // switch to alternate version, namespaced to new dependent since we don't know the version is cluster-scoped
		},
		{
			name: "cluster alternate kind",
			deps: []*node{
				makeNode(ns1child1, withOwners(clusterabsentparent)),
				makeNode(ns1child2, withOwners(clusterabsentparent_kind)),
			},
			verifiedAbsent:    clusterabsentparent,
			expectedAlternate: &clusterabsentparent_ns1_kind, // switch to alternate kind, namespaced to new dependent since we don't know the new kind is cluster-scoped
		},
		{
			name: "cluster alternate namespace",
			deps: []*node{
				makeNode(ns1child1, withOwners(clusterabsentparent)),
				makeNode(ns2child1, withOwners(clusterabsentparent)),
			},
			verifiedAbsent:    clusterabsentparent,
			expectedAlternate: nil, // apiVersion/kind verified cluster-scoped, namespace delta ignored, no alternates found
		},
		{
			name: "cluster alternate name",
			deps: []*node{
				makeNode(ns1child1, withOwners(clusterabsentparent)),
				makeNode(ns1child1, withOwners(clusterabsentparent_name)),
			},
			verifiedAbsent:    clusterabsentparent,
			expectedAlternate: &clusterabsentparent_name, // switch to alternate name, apiVersion/kind verified cluster-scoped, namespace dropped
		},

		{
			name: "namespaced ref from namespaced child returns first if absent is sorted last",
			deps: []*node{
				makeNode(ns1child1, withOwners(orderedNamespacedReferences...)),
			},
			verifiedAbsent:    orderedNamespacedReferences[len(orderedNamespacedReferences)-1],
			expectedAlternate: &orderedNamespacedReferences[0],
		},
		{
			name: "namespaced ref from namespaced child returns next after absent",
			deps: []*node{
				makeNode(ns1child1, withOwners(orderedNamespacedReferences...)),
			},
			verifiedAbsent:    orderedNamespacedReferences[len(orderedNamespacedReferences)-2],
			expectedAlternate: &orderedNamespacedReferences[len(orderedNamespacedReferences)-1],
		},

		{
			name: "cluster ref from cluster child returns first if absent is sorted last",
			deps: []*node{
				makeNode(clusterchild1, withOwners(orderedClusterReferences...)),
			},
			verifiedAbsent:    orderedClusterReferences[len(orderedClusterReferences)-1],
			expectedAlternate: &orderedClusterReferences[0],
		},
		{
			name: "cluster ref from cluster child returns next after absent",
			deps: []*node{
				makeNode(clusterchild1, withOwners(orderedClusterReferences...)),
			},
			verifiedAbsent:    orderedClusterReferences[len(orderedClusterReferences)-2],
			expectedAlternate: &orderedClusterReferences[len(orderedClusterReferences)-1],
		},

		{
			name: "ignore unrelated",
			deps: []*node{
				makeNode(ns1child1, withOwners(clusterabsentparent, makeID("v1", "Parent", "ns1", "name", "anotheruid"))),
			},
			verifiedAbsent:    clusterabsentparent,
			expectedAlternate: nil,
		},
		{
			name: "ignore matches",
			deps: []*node{
				makeNode(ns1child1, withOwners(clusterabsentparent, clusterabsentparent)),
			},
			verifiedAbsent:    clusterabsentparent,
			expectedAlternate: nil,
		},
		{
			name: "collapse duplicates",
			deps: []*node{
				makeNode(clusterchild1, withOwners(clusterabsentparent, clusterabsentparent_kind, clusterabsentparent_kind)),
			},
			verifiedAbsent:    clusterabsentparent,
			expectedAlternate: &clusterabsentparent_kind,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			alternate := getAlternateOwnerIdentity(tc.deps, tc.verifiedAbsent)
			if !reflect.DeepEqual(alternate, tc.expectedAlternate) {
				t.Errorf("expected\n%#v\ngot\n%#v", tc.expectedAlternate, alternate)
			}
		})
	}
}
