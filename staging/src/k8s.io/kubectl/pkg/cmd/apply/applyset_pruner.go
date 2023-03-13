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

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/dynamic"
	"k8s.io/klog/v2"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

type applySetPruner struct {
	dynamicClient dynamic.Interface

	visitedUids sets.Set[types.UID]

	cascadingStrategy metav1.DeletionPropagation
	dryRunStrategy    cmdutil.DryRunStrategy
	gracePeriod       int

	printer printers.ResourcePrinter

	ioStreams genericclioptions.IOStreams
}

func newApplySetPruner(o *ApplyOptions) (*applySetPruner, error) {
	printer, err := o.ToPrinter("pruned")
	if err != nil {
		return nil, err
	}

	return &applySetPruner{
		dynamicClient: o.DynamicClient,

		visitedUids: o.VisitedUids,

		cascadingStrategy: o.DeleteOptions.CascadingStrategy,
		dryRunStrategy:    o.DryRunStrategy,
		gracePeriod:       o.DeleteOptions.GracePeriod,

		printer: printer,

		ioStreams: o.IOStreams,
	}, nil
}

func (p *applySetPruner) pruneAll(ctx context.Context, applyset *ApplySet) error {
	// TODO: Split into discovery and deletion, run discovery in parallel (and maybe in consistent order or in parallel?)
	for _, restMapping := range applyset.AllPrunableResources() {
		switch restMapping.Scope.Name() {
		case meta.RESTScopeNameNamespace:
			for _, namespace := range applyset.AllPrunableNamespaces() {
				if namespace == "" {
					// Just double-check because otherwise we get cryptic error messages
					return fmt.Errorf("unexpectedly encountered empty namespace during prune of namespace-scoped resource %v", restMapping.GroupVersionKind)
				}
				if err := p.prune(ctx, namespace, restMapping, applyset); err != nil {
					return fmt.Errorf("pruning %v objects: %w", restMapping.GroupVersionKind.String(), err)
				}
			}

		case meta.RESTScopeNameRoot:
			if err := p.prune(ctx, metav1.NamespaceNone, restMapping, applyset); err != nil {
				return fmt.Errorf("pruning %v objects: %w", restMapping.GroupVersionKind.String(), err)
			}

		default:
			return fmt.Errorf("unhandled scope %q", restMapping.Scope.Name())
		}
	}

	return nil
}

func (p *applySetPruner) prune(ctx context.Context, namespace string, mapping *meta.RESTMapping, applyset *ApplySet) error {
	applysetLabelSelector := applyset.LabelSelectorForMembers()

	opt := metav1.ListOptions{
		LabelSelector: applysetLabelSelector,
	}

	klog.V(2).Infof("listing objects for pruning; namespace=%q, resource=%v", namespace, mapping.Resource)
	objects, err := p.dynamicClient.Resource(mapping.Resource).Namespace(namespace).List(ctx, opt)
	if err != nil {
		return err
	}

	for i := range objects.Items {
		obj := &objects.Items[i]

		uid := obj.GetUID()
		if p.visitedUids.Has(uid) {
			continue
		}
		name := obj.GetName()
		if p.dryRunStrategy != cmdutil.DryRunClient {
			if err := p.delete(ctx, namespace, name, mapping); err != nil {
				return fmt.Errorf("deleting %s/%s: %w", namespace, name, err)
			}
		}

		p.printer.PrintObj(obj, p.ioStreams.Out)
	}
	return nil
}

func (p *applySetPruner) delete(ctx context.Context, namespace string, name string, mapping *meta.RESTMapping) error {
	return runDelete(ctx, namespace, name, mapping, p.dynamicClient, p.cascadingStrategy, p.gracePeriod, p.dryRunStrategy == cmdutil.DryRunServer)
}
