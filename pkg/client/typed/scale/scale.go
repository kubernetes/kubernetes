/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package scale

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	unversionedclient "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
)

type Client struct {
	dc *dynamic.Client
}

type ScaleInterface interface {
	Replicas() (int, error)
	SetReplicas(replicas int)
	Selector() (string, error)
}

func NewClient(c *unversionedclient.Config) (*Client, error) {
	dc, err := dynamic.NewClient(c)
	if err != nil {
		return nil, err
	}
	return &Client{
		dc: dc,
	}, nil
}

// NewForConfigOrDie creates a new dynamic client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *unversionedclient.Config) *Client {
	cl, err := NewClient(c)
	if err != nil {
		panic(err)
	}
	return cl
}

func (c *Client) Get(scaleRef extensions.SubresourceReference, namespace string) (ScaleInterface, error) {
	gv, err := unversioned.ParseGroupVersion(scaleRef.APIVersion)
	if err != nil {
		return nil, fmt.Errorf("error parsing group version from scale reference: %v", err)
	}
	gvr, _ := meta.KindToResource(gv.WithKind(scaleRef.Kind))

	resource := &unversioned.APIResource{
		Name:       gvr.Resource,
		Namespaced: true,
		Kind:       scaleRef.Kind,
	}
	res, err := c.dc.Subresource(resource, scaleRef.Subresource, namespace).Get(scaleRef.Name)
	if err != nil {
		return nil, fmt.Errorf("unable to fetch resource %s(%s): %v", scaleRef.Name, scaleRef.Kind, err)
	}
	if res.TypeMeta.Kind == "ReplicationController" && res.TypeMeta.APIVersion == "extensions/v1beta" {
		return &extScale{scale{c, res}}, nil
	} else {
		return &asScale{scale{c, res}}, nil
	}
}

func (c *Client) Update(scaleRef extensions.SubresourceReference, namespace string, obj ScaleInterface) (ScaleInterface, error) {
	gv, err := unversioned.ParseGroupVersion(scaleRef.APIVersion)
	if err != nil {
		return nil, fmt.Errorf("error parsing group version from scale reference: %v", err)
	}
	gvr, _ := meta.KindToResource(gv.WithKind(scaleRef.Kind))

	resource := &unversioned.APIResource{
		Name:       gvr.Resource,
		Namespaced: true,
		Kind:       scaleRef.Kind,
	}
	var res *runtime.Unstructured
	switch t := obj.(type) {
	case *extScale:
		res, err = c.dc.Subresource(resource, scaleRef.Subresource, namespace).Update(t.getScale())
	case *asScale:
		res, err = c.dc.Subresource(resource, scaleRef.Subresource, namespace).Update(t.getScale())
	default:
		return nil, fmt.Errorf("error obtaining existing scale values")
	}
	if err != nil {
		return nil, fmt.Errorf("unable to fetch resource %s(%s): %v", scaleRef.Name, scaleRef.Kind, err)
	}

	if res.TypeMeta.Kind == "ReplicationController" && res.TypeMeta.APIVersion == "extensions/v1beta" {
		return &extScale{scale{c, res}}, nil
	} else {
		return &asScale{scale{c, res}}, nil
	}
}

func field(obj map[string]interface{}, name string) (interface{}, error) {
	val, ok := obj[name]
	if !ok {
		return nil, fmt.Errorf("`%s` field doesn't exist in the `Scale` object", name)
	}
	return val, nil
}

type scale struct {
	c   *Client
	obj *runtime.Unstructured
}

func (s *scale) getScale() *runtime.Unstructured {
	return s.obj
}

type extScale struct {
	scale
}

func (s *extScale) Replicas() (int, error) {
	statusVal, err := field(s.obj.Object, "Status")
	if err != nil {
		return 0, err
	}
	status, ok := statusVal.(*extensions.ScaleStatus)
	if !ok {
		return 0, fmt.Errorf("`Status` isn't of expected type")
	}
	return status.Replicas, nil
}

// TODO: Implementing this as a Get-Update cycle for now, but it should really be
// implemented as a Patch.
func (s *extScale) SetReplicas(replicas int) {
	s.obj.Object["Spec"] = &extensions.ScaleSpec{replicas}
}

func (s *extScale) Selector() (string, error) {
	statusVal, err := field(s.obj.Object, "Status")
	if err != nil {
		return "", err
	}
	status, ok := statusVal.(*extensions.ScaleStatus)
	if !ok {
		return "", fmt.Errorf("`Status` isn't of expected type")
	}
	return labels.SelectorFromSet(status.Selector).String(), nil
}

type asScale struct {
	scale
}

func (s *asScale) Replicas() (int, error) {
	statusVal, err := field(s.obj.Object, "Status")
	if err != nil {
		return 0, err
	}
	status, ok := statusVal.(*autoscaling.ScaleStatus)
	if !ok {
		return 0, fmt.Errorf("`Status` isn't of expected type")
	}
	return status.Replicas, nil
}

// TODO: Implementing this as a Get-Update cycle for now, but it should really be
// implemented as a Patch.
func (s *asScale) SetReplicas(replicas int) {
	s.obj.Object["Spec"] = &extensions.ScaleSpec{replicas}
}

func (s *asScale) Selector() (string, error) {
	statusVal, err := field(s.obj.Object, "Status")
	if err != nil {
		return "", err
	}
	status, ok := statusVal.(*autoscaling.ScaleStatus)
	if !ok {
		return "", fmt.Errorf("`Status` isn't of expected type")
	}
	return status.Selector, nil
}
