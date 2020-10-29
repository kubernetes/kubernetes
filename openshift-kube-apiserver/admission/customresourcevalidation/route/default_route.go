package route

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"

	v1 "github.com/openshift/api/route/v1"
)

const (
	DefaultingPluginName = "route.openshift.io/DefaultRoute"
)

func RegisterDefaulting(plugins *admission.Plugins) {
	plugins.Register(DefaultingPluginName, func(_ io.Reader) (admission.Interface, error) {
		return &defaultRoute{
			Handler: admission.NewHandler(admission.Create, admission.Update),
		}, nil
	})
}

type defaultRoute struct {
	*admission.Handler
}

var _ admission.MutationInterface = &defaultRoute{}

func (a *defaultRoute) Admit(ctx context.Context, attributes admission.Attributes, _ admission.ObjectInterfaces) error {
	if attributes.GetResource().GroupResource() != (schema.GroupResource{Group: "route.openshift.io", Resource: "routes"}) {
		return nil
	}

	if len(attributes.GetSubresource()) > 0 {
		return nil
	}

	u, ok := attributes.GetObject().(runtime.Unstructured)
	if !ok {
		// If a request to the resource routes.route.openshift.io is subject to
		// kube-apiserver admission, that should imply that the route API is being served as
		// CRs and the request body should have been unmarshaled into an unstructured
		// object.
		return fmt.Errorf("object being admitted is of type %T and does not implement runtime.Unstructured", attributes.GetObject())
	}

	var external v1.Route
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(u.UnstructuredContent(), &external); err != nil {
		return err
	}

	SetObjectDefaults_Route(&external)

	content, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&external)
	if err != nil {
		return err
	}
	u.SetUnstructuredContent(content)

	return nil
}
