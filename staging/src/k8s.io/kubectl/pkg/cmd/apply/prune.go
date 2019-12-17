/*
Copyright 2019 The Kubernetes Authors.

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

package apply

import (
	"fmt"
	"io"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/dynamic"
)

type pruner struct {
	mapper        meta.RESTMapper
	dynamicClient dynamic.Interface

	visitedUids       sets.String
	visitedNamespaces sets.String
	labelSelector     string
	fieldSelector     string

	cascade      bool
	serverDryRun bool
	dryRun       bool
	gracePeriod  int

	toPrinter func(string) (printers.ResourcePrinter, error)

	out io.Writer
}

func newPruner(o *ApplyOptions) pruner {
	return pruner{
		mapper:        o.Mapper,
		dynamicClient: o.DynamicClient,

		labelSelector:     o.Selector,
		visitedUids:       o.VisitedUids,
		visitedNamespaces: o.VisitedNamespaces,

		cascade:      o.DeleteOptions.Cascade,
		dryRun:       o.DryRun,
		serverDryRun: o.ServerDryRun,
		gracePeriod:  o.DeleteOptions.GracePeriod,

		toPrinter: o.ToPrinter,

		out: o.Out,
	}
}

func (p *pruner) pruneAll(o *ApplyOptions) error {

	namespacedRESTMappings, nonNamespacedRESTMappings, err := getRESTMappings(o.Mapper, &(o.PruneResources))
	if err != nil {
		return fmt.Errorf("error retrieving RESTMappings to prune: %v", err)
	}

	for n := range p.visitedNamespaces {
		if len(o.Namespace) != 0 && n != o.Namespace {
			continue
		}
		for _, m := range namespacedRESTMappings {
			if err := p.prune(n, m); err != nil {
				return fmt.Errorf("error pruning namespaced object %v: %v", m.GroupVersionKind, err)
			}
		}
	}
	for _, m := range nonNamespacedRESTMappings {
		if err := p.prune(metav1.NamespaceNone, m); err != nil {
			return fmt.Errorf("error pruning nonNamespaced object %v: %v", m.GroupVersionKind, err)
		}
	}

	return nil
}

func (p *pruner) prune(namespace string, mapping *meta.RESTMapping) error {
	objList, err := p.dynamicClient.Resource(mapping.Resource).
		Namespace(namespace).
		List(metav1.ListOptions{
			LabelSelector: p.labelSelector,
			FieldSelector: p.fieldSelector,
		})
	if err != nil {
		return err
	}

	objs, err := meta.ExtractList(objList)
	if err != nil {
		return err
	}

	for _, obj := range objs {
		metadata, err := meta.Accessor(obj)
		if err != nil {
			return err
		}
		annots := metadata.GetAnnotations()
		if _, ok := annots[corev1.LastAppliedConfigAnnotation]; !ok {
			// don't prune resources not created with apply
			continue
		}
		uid := metadata.GetUID()
		if p.visitedUids.Has(string(uid)) {
			continue
		}
		name := metadata.GetName()
		if !p.dryRun {
			if err := p.delete(namespace, name, mapping); err != nil {
				return err
			}
		}

		printer, err := p.toPrinter("pruned")
		if err != nil {
			return err
		}
		printer.PrintObj(obj, p.out)
	}
	return nil
}

func (p *pruner) delete(namespace, name string, mapping *meta.RESTMapping) error {
	return runDelete(namespace, name, mapping, p.dynamicClient, p.cascade, p.gracePeriod, p.serverDryRun)
}

func runDelete(namespace, name string, mapping *meta.RESTMapping, c dynamic.Interface, cascade bool, gracePeriod int, serverDryRun bool) error {
	options := &metav1.DeleteOptions{}
	if gracePeriod >= 0 {
		options = metav1.NewDeleteOptions(int64(gracePeriod))
	}
	if serverDryRun {
		options.DryRun = []string{metav1.DryRunAll}
	}
	policy := metav1.DeletePropagationForeground
	if !cascade {
		policy = metav1.DeletePropagationOrphan
	}
	options.PropagationPolicy = &policy
	return c.Resource(mapping.Resource).Namespace(namespace).Delete(name, options)
}

type pruneResource struct {
	group      string
	version    string
	kind       string
	namespaced bool
}

func (pr pruneResource) String() string {
	return fmt.Sprintf("%v/%v, Kind=%v, Namespaced=%v", pr.group, pr.version, pr.kind, pr.namespaced)
}

func getRESTMappings(mapper meta.RESTMapper, pruneResources *[]pruneResource) (namespaced, nonNamespaced []*meta.RESTMapping, err error) {
	if len(*pruneResources) == 0 {
		// default whitelist
		// TODO: need to handle the older api versions - e.g. v1beta1 jobs. Github issue: #35991
		*pruneResources = []pruneResource{
			{"", "v1", "ConfigMap", true},
			{"", "v1", "Endpoints", true},
			{"", "v1", "Namespace", false},
			{"", "v1", "PersistentVolumeClaim", true},
			{"", "v1", "PersistentVolume", false},
			{"", "v1", "Pod", true},
			{"", "v1", "ReplicationController", true},
			{"", "v1", "Secret", true},
			{"", "v1", "Service", true},
			{"batch", "v1", "Job", true},
			{"batch", "v1beta1", "CronJob", true},
			{"extensions", "v1beta1", "Ingress", true},
			{"apps", "v1", "DaemonSet", true},
			{"apps", "v1", "Deployment", true},
			{"apps", "v1", "ReplicaSet", true},
			{"apps", "v1", "StatefulSet", true},
		}
	}

	for _, resource := range *pruneResources {
		addedMapping, err := mapper.RESTMapping(schema.GroupKind{Group: resource.group, Kind: resource.kind}, resource.version)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid resource %v: %v", resource, err)
		}
		if resource.namespaced {
			namespaced = append(namespaced, addedMapping)
		} else {
			nonNamespaced = append(nonNamespaced, addedMapping)
		}
	}

	return namespaced, nonNamespaced, nil
}

func parsePruneResources(mapper meta.RESTMapper, gvks []string) ([]pruneResource, error) {
	pruneResources := []pruneResource{}
	for _, groupVersionKind := range gvks {
		gvk := strings.Split(groupVersionKind, "/")
		if len(gvk) != 3 {
			return nil, fmt.Errorf("invalid GroupVersionKind format: %v, please follow <group/version/kind>", groupVersionKind)
		}

		if gvk[0] == "core" {
			gvk[0] = ""
		}
		mapping, err := mapper.RESTMapping(schema.GroupKind{Group: gvk[0], Kind: gvk[2]}, gvk[1])
		if err != nil {
			return pruneResources, err
		}
		var namespaced bool
		namespaceScope := mapping.Scope.Name()
		switch namespaceScope {
		case meta.RESTScopeNameNamespace:
			namespaced = true
		case meta.RESTScopeNameRoot:
			namespaced = false
		default:
			return pruneResources, fmt.Errorf("Unknown namespace scope: %q", namespaceScope)
		}

		pruneResources = append(pruneResources, pruneResource{gvk[0], gvk[1], gvk[2], namespaced})
	}
	return pruneResources, nil
}
