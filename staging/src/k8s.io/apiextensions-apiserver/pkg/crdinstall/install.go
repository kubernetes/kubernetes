package crdinstall

import (
	"context"
	k8sapierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"time"
)

func Install(ctx context.Context, config *rest.Config) error {
	// wait for Kind: CustomResourceDefinition to become available
	// TODO: replace this wait with a more standard approach where Install is
	// only called after CRDs are available
	time.Sleep(20 * time.Second)
	objs, err := getObjects()
	if err != nil {
		return err
	}

	for _, obj := range objs {
		if err := installOne(ctx, config, obj); err != nil {
			return err
		}
	}
	return nil
}

func installOne(ctx context.Context, config *rest.Config, obj *unstructured.Unstructured) error {
	if err := getResource(ctx, config, obj); err != nil {
		if k8sapierrors.IsNotFound(err) {
			// The resource does not exist on the cluster. Create it.
			if err := createResource(ctx, config, obj); err != nil {
				return err
			}
			return nil
		}
		return err
	} else {
		// The resource already exists on the cluster. Patch it.
		if err := patchResource(ctx, config, obj); err != nil {
			return err
		}
	}
	return nil
}

func getResource(ctx context.Context, config *rest.Config, obj *unstructured.Unstructured) error {
	dc, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return err
	}
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(memory.NewMemCacheClient(dc))

	gvk := obj.GroupVersionKind()
	mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return err
	}

	dyn, err := dynamic.NewForConfig(config)
	if err != nil {
		return err
	}

	var dynRes dynamic.ResourceInterface
	if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
		dynRes = dyn.Resource(mapping.Resource).Namespace(obj.GetNamespace())
	} else {
		dynRes = dyn.Resource(mapping.Resource)
	}

	_, err = dynRes.Get(ctx, obj.GetName(), metav1.GetOptions{})
	if err != nil {
		return err
	}

	return nil
}

func createResource(ctx context.Context, config *rest.Config, obj *unstructured.Unstructured) error {
	dc, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return err
	}
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(memory.NewMemCacheClient(dc))

	gvk := obj.GroupVersionKind()
	mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return err
	}

	dyn, err := dynamic.NewForConfig(config)
	if err != nil {
		return err
	}

	var dynRes dynamic.ResourceInterface
	if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
		dynRes = dyn.Resource(mapping.Resource).Namespace(obj.GetNamespace())
	} else {
		dynRes = dyn.Resource(mapping.Resource)
	}

	_, err = dynRes.Create(ctx, obj, metav1.CreateOptions{FieldManager: "awesome-client"})
	if err != nil {
		return err
	}

	return nil
}

func patchResource(ctx context.Context, config *rest.Config, obj *unstructured.Unstructured) error {
	dc, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return err
	}
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(memory.NewMemCacheClient(dc))

	gvk := obj.GroupVersionKind()
	mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return err
	}

	dyn, err := dynamic.NewForConfig(config)
	if err != nil {
		return err
	}

	var dynRes dynamic.ResourceInterface
	if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
		dynRes = dyn.Resource(mapping.Resource).Namespace(obj.GetNamespace())
	} else {
		dynRes = dyn.Resource(mapping.Resource)
	}

	data, err := obj.MarshalJSON()
	if err != nil {
		return err
	}

	_, err = dynRes.Patch(ctx, obj.GetName(), types.JSONPatchType, data, metav1.PatchOptions{FieldManager: "awesome-client"})
	if err != nil {
		return err
	}

	return nil
}
