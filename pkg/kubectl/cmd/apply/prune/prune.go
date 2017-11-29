/*
Copyright 2014 The Kubernetes Authors.

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

package prune

import (
	"fmt"
	"io"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

type Pruner struct {
	Mapper               meta.RESTMapper
	GroupVersionKinds    []*meta.RESTMapping
	ClientFunc           resource.ClientMapperFunc
	ClientsetFunc        func() (internalclientset.Interface, error)
	Selector             string
	IncludeUninitialized bool
	Cascade              bool
	GracePeriod          int
	DryRun               bool
	Out                  io.Writer
	ShortOutput          bool
}

// Param: gvks array of groupVersionKind passed in --prune-whitelist or empty array if no flag.
// Returns default groupVersionKind mappings to prune, unless overriden by --prune-whitelist flag.
func ParseGvks(mapper meta.RESTMapper, gvks []string) ([]*meta.RESTMapping, error) {

	// Default group/version/kind to prune
	pruneGvks := [][]string{
		{"", "v1", "ConfigMap"},
		{"", "v1", "Endpoints"},
		{"", "v1", "Namespace"},
		{"", "v1", "PersistentVolumeClaim"},
		{"", "v1", "PersistentVolume"},
		{"", "v1", "Pod"},
		{"", "v1", "ReplicationController"},
		{"", "v1", "Secret"},
		{"", "v1", "Service"},
		{"batch", "v1", "Job"},
		{"extensions", "v1beta1", "DaemonSet"},
		{"extensions", "v1beta1", "Deployment"},
		{"extensions", "v1beta1", "Ingress"},
		{"extensions", "v1beta1", "ReplicaSet"},
		{"apps", "v1beta1", "StatefulSet"},
		{"apps", "v1beta1", "Deployment"},
	}

	// If group/version/kind specified in "prune-whitelist" flag, then use those values
	if len(gvks) != 0 {
		var err error
		pruneGvks, err = parsePruneWhitelist(gvks)
		if err != nil {
			return nil, err
		}
	}

	// Parse groupVersionKind from passed prune-whitelist flag values
	var groupVersionKinds []*meta.RESTMapping
	for _, gvk := range pruneGvks {
		mapping, err := mapper.RESTMapping(schema.GroupKind{Group: gvk[0], Kind: gvk[2]}, gvk[1])
		if err != nil {
			return nil, err
		}
		groupVersionKinds = append(groupVersionKinds, mapping)
	}

	return groupVersionKinds, nil
}

// Example gvks: "core/v1/endpoints", "apps/v1beta1/deployment"
// Returns validated gvks as array of arrays of strings
func parsePruneWhitelist(gvks []string) ([][]string, error) {

	var parsedGvks [][]string
	for _, groupVersionKind := range gvks {
		gvk := strings.Split(groupVersionKind, "/")
		if len(gvk) != 3 {
			return nil, fmt.Errorf("invalid GroupVersionKind format: %v, please follow <group/version/kind>", groupVersionKind)
		}
		if gvk[0] == "core" {
			gvk[0] = ""
		}
		parsedGvks = append(parsedGvks, gvk)
	}
	return parsedGvks, nil
}

// Prune resources not previously visited.
func (p *Pruner) Prune(visitedNamespaces sets.String, visitedUids sets.String) error {
	// Prune namespaced resources
	for n := range visitedNamespaces {
		for _, m := range p.GroupVersionKinds {
			if m.Scope.Name() == meta.RESTScopeNameNamespace {
				if err := p.pruneMapping(n, m, visitedUids); err != nil {
					return fmt.Errorf("error pruning namespaced object %v: %v", m.GroupVersionKind, err)
				}
			}
		}
	}
	// Prune non-namespaced resources
	for _, m := range p.GroupVersionKinds {
		if m.Scope.Name() == meta.RESTScopeNameRoot {
			if err := p.pruneMapping(metav1.NamespaceNone, m, visitedUids); err != nil {
				return fmt.Errorf("error pruning non-namespaced object %v: %v", m.GroupVersionKind, err)
			}
		}
	}
	return nil
}

// Deletes objects in namespace/group/version/kind which have not previously been visited.
func (p *Pruner) pruneMapping(namespace string, mapping *meta.RESTMapping, visitedUids sets.String) error {
	c, err := p.ClientFunc(mapping)
	if err != nil {
		return err
	}

	export := false
	apiVersion := mapping.GroupVersionKind.Version
	objList, err := resource.NewHelper(c, mapping).List(
		namespace,
		apiVersion,
		export,
		&metav1.ListOptions{
			LabelSelector:        p.Selector,
			IncludeUninitialized: p.IncludeUninitialized,
		},
	)
	if err != nil {
		return err
	}
	objs, err := meta.ExtractList(objList)
	if err != nil {
		return err
	}

	for _, obj := range objs {
		annots, err := mapping.MetadataAccessor.Annotations(obj)
		if err != nil {
			return err
		}
		if _, ok := annots[api.LastAppliedConfigAnnotation]; !ok {
			// don't prune resources not created with apply
			continue
		}
		uid, err := mapping.UID(obj)
		if err != nil {
			return err
		}
		if visitedUids.Has(string(uid)) {
			continue
		}

		name, err := mapping.Name(obj)
		if err != nil {
			return err
		}
		if !p.DryRun {
			if err := p.delete(namespace, name, mapping); err != nil {
				return err
			}
		}
		p.printSuccess(mapping.Resource, name, "pruned")
	}
	return nil
}

func (p *Pruner) delete(namespace, name string, mapping *meta.RESTMapping) error {

	c, err := p.ClientFunc(mapping)
	if err != nil {
		return err
	}

	helper := resource.NewHelper(c, mapping)
	if !p.Cascade {
		return helper.Delete(namespace, name)
	}

	cs, err := p.ClientsetFunc()
	if err != nil {
		return err
	}
	r, err := kubectl.ReaperFor(mapping.GroupVersionKind.GroupKind(), cs)
	if err != nil {
		if _, ok := err.(*kubectl.NoSuchReaperError); !ok {
			return err
		}
		return helper.Delete(namespace, name)
	}
	var options *metav1.DeleteOptions
	if p.GracePeriod >= 0 {
		options = metav1.NewDeleteOptions(int64(p.GracePeriod))
	}
	stopTimeout := 2 * time.Minute
	if err := r.Stop(namespace, name, stopTimeout, options); err != nil {
		return err
	}
	return nil
}

func (p *Pruner) printSuccess(resource, name string, operation string) {
	resource, _ = p.Mapper.ResourceSingularizer(resource)
	dryRunMsg := ""
	if p.DryRun {
		dryRunMsg = " (dry run)"
	}
	if p.ShortOutput {
		// -o name: prints resource/name
		if len(resource) > 0 {
			fmt.Fprintf(p.Out, "%s/%s\n", resource, name)
		} else {
			fmt.Fprintf(p.Out, "%s\n", name)
		}
	} else {
		// understandable output by default
		if len(resource) > 0 {
			fmt.Fprintf(p.Out, "%s \"%s\" %s%s\n", resource, name, operation, dryRunMsg)
		} else {
			fmt.Fprintf(p.Out, "\"%s\" %s%s\n", name, operation, dryRunMsg)
		}
	}
}
