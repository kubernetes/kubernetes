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

package apply

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/dynamic"
	"k8s.io/klog/v2"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

type ApplySetDeleteOptions struct {
	CascadingStrategy metav1.DeletionPropagation
	DryRunStrategy    cmdutil.DryRunStrategy
	GracePeriod       int

	Printer printers.ResourcePrinter

	IOStreams genericiooptions.IOStreams
}

// PruneObject is an apiserver object that should be deleted as part of prune.
type PruneObject struct {
	Name      string
	Namespace string
	Mapping   *meta.RESTMapping
	Object    runtime.Object
}

// String returns a human-readable name of the object, for use in debug messages.
func (p *PruneObject) String() string {
	s := p.Mapping.GroupVersionKind.GroupKind().String()

	if p.Namespace != "" {
		s += " " + p.Namespace + "/" + p.Name
	} else {
		s += " " + p.Name
	}
	return s
}

// FindAllObjectsToPrune returns the list of objects that will be pruned.
// Calling this instead of Prune can be useful for dry-run / diff behaviour.
func (a *ApplySet) FindAllObjectsToPrune(ctx context.Context, dynamicClient dynamic.Interface, visitedUids sets.Set[types.UID]) ([]PruneObject, error) {
	type task struct {
		namespace   string
		restMapping *meta.RESTMapping

		err     error
		results []PruneObject
	}
	var tasks []*task

	// We run discovery in parallel, in as many goroutines as priority and fairness will allow
	// (We don't expect many requests in real-world scenarios - maybe tens, unlikely to be hundreds)
	for gvk, resource := range a.AllPrunableResources() {
		scope := resource.restMapping.Scope

		switch scope.Name() {
		case meta.RESTScopeNameNamespace:
			for _, namespace := range a.AllPrunableNamespaces() {
				if namespace == "" {
					// Just double-check because otherwise we get cryptic error messages
					return nil, fmt.Errorf("unexpectedly encountered empty namespace during prune of namespace-scoped resource %v", gvk)
				}
				tasks = append(tasks, &task{
					namespace:   namespace,
					restMapping: resource.restMapping,
				})
			}

		case meta.RESTScopeNameRoot:
			tasks = append(tasks, &task{
				restMapping: resource.restMapping,
			})

		default:
			return nil, fmt.Errorf("unhandled scope %q", scope.Name())
		}
	}

	var wg sync.WaitGroup

	for i := range tasks {
		task := tasks[i]
		wg.Add(1)
		go func() {
			defer wg.Done()

			results, err := a.findObjectsToPrune(ctx, dynamicClient, visitedUids, task.namespace, task.restMapping)
			if err != nil {
				task.err = fmt.Errorf("listing %v objects for pruning: %w", task.restMapping.GroupVersionKind.String(), err)
			} else {
				task.results = results
			}
		}()
	}
	// Wait for all the goroutines to finish
	wg.Wait()

	var allObjects []PruneObject
	for _, task := range tasks {
		if task.err != nil {
			return nil, task.err
		}
		allObjects = append(allObjects, task.results...)
	}
	return allObjects, nil
}

func (a *ApplySet) pruneAll(ctx context.Context, dynamicClient dynamic.Interface, visitedUids sets.Set[types.UID], deleteOptions *ApplySetDeleteOptions) error {
	allObjects, err := a.FindAllObjectsToPrune(ctx, dynamicClient, visitedUids)
	if err != nil {
		return err
	}

	return a.deleteObjects(ctx, dynamicClient, allObjects, deleteOptions)
}

func (a *ApplySet) findObjectsToPrune(ctx context.Context, dynamicClient dynamic.Interface, visitedUids sets.Set[types.UID], namespace string, mapping *meta.RESTMapping) ([]PruneObject, error) {
	applysetLabelSelector := a.LabelSelectorForMembers()

	opt := metav1.ListOptions{
		LabelSelector: applysetLabelSelector,
	}

	klog.V(2).Infof("listing objects for pruning; namespace=%q, resource=%v", namespace, mapping.Resource)
	objects, err := dynamicClient.Resource(mapping.Resource).Namespace(namespace).List(ctx, opt)
	if err != nil {
		return nil, err
	}

	var pruneObjects []PruneObject
	for i := range objects.Items {
		obj := &objects.Items[i]

		uid := obj.GetUID()
		if visitedUids.Has(uid) {
			continue
		}
		name := obj.GetName()
		pruneObjects = append(pruneObjects, PruneObject{
			Name:      name,
			Namespace: namespace,
			Mapping:   mapping,
			Object:    obj,
		})

	}
	return pruneObjects, nil
}

func (a *ApplySet) deleteObjects(ctx context.Context, dynamicClient dynamic.Interface, pruneObjects []PruneObject, opt *ApplySetDeleteOptions) error {
	for i := range pruneObjects {
		pruneObject := &pruneObjects[i]

		name := pruneObject.Name
		namespace := pruneObject.Namespace
		mapping := pruneObject.Mapping

		if opt.DryRunStrategy != cmdutil.DryRunClient {
			if err := runDelete(ctx, namespace, name, mapping, dynamicClient, opt.CascadingStrategy, opt.GracePeriod, opt.DryRunStrategy == cmdutil.DryRunServer); err != nil {
				return fmt.Errorf("pruning %v: %w", pruneObject.String(), err)
			}
		}

		opt.Printer.PrintObj(pruneObject.Object, opt.IOStreams.Out)

	}
	return nil
}
