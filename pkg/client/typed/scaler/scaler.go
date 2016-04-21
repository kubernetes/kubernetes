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

// Package scaler defines and implements an interface polymorphic consumption
// of the scale subresource.
package scaler

import (
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
)

type Client struct {
	coredc *dynamic.Client
	extdc  *dynamic.Client
}

// ScalerInterface defines a set of methods required to consume the Scale subresource
// irrespective of the concrete Scale API type.
type ScalerInterface interface {
	getScale() *runtime.Unstructured
	Replicas() (int, error)
	SetReplicas(replicas int)
	Selector() (labels.Selector, error)
}

// NewClient returns a new client to operate on the scale subresource. The returned client
// is backed by a dynamic client which uses its own codec.
func NewClient(conf *restclient.Config) (*Client, error) {
	// conf is passed as a pointer, do not modify it
	coreConf := *conf
	if coreConf.APIPath == "" {
		coreConf.APIPath = "/api"
	}
	coreConf.GroupVersion = &unversioned.GroupVersion{
		Group:   "",
		Version: "v1",
	}
	coredc, err := dynamic.NewClient(&coreConf)
	if err != nil {
		return nil, err
	}

	extConf := *conf
	if extConf.APIPath == "" {
		extConf.APIPath = "/apis"
	}
	extConf.GroupVersion = &unversioned.GroupVersion{
		Group:   "extensions",
		Version: "v1beta1",
	}
	extdc, err := dynamic.NewClient(&extConf)
	if err != nil {
		return nil, err
	}

	return &Client{
		coredc: coredc,
		extdc:  extdc,
	}, nil
}

// NewForConfigOrDie creates a new scale client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(conf *restclient.Config) *Client {
	cl, err := NewClient(conf)
	if err != nil {
		panic(err)
	}
	return cl
}

func (c *Client) client(gv unversioned.GroupVersion) *dynamic.Client {
	if len(gv.Group) == 0 && gv.Version == "v1" {
		return c.coredc
	}
	return c.extdc
}

// TODO(madhusudancs): Implement this as a codec decoder.
func (c *Client) decode(res *runtime.Unstructured) (ScalerInterface, error) {
	if res.TypeMeta.Kind != "Scale" {
		return nil, fmt.Errorf("object should be of Kind `Scale`, got `%s`", res.TypeMeta.Kind)
	}
	switch res.TypeMeta.APIVersion {
	case "extensions/v1beta1":
		return &extScale{scale{c, res}}, nil
	case "autoscaling/v1":
		return &asScale{scale{c, res}}, nil
	default:
		return nil, fmt.Errorf("unknown APIVersion `%s` for `Scale`", res.TypeMeta.APIVersion)
	}
}

func (c *Client) Get(scaleRef extensions.SubresourceReference, namespace string) (ScalerInterface, error) {
	gv, err := unversioned.ParseGroupVersion(scaleRef.APIVersion)
	if err != nil {
		return nil, fmt.Errorf("error parsing group version from scale reference: %v", err)
	}

	// Second return value isn't an error here, it is just the singular version of
	// of the resource name. Since we don't care about the singular version in our
	// case we can just ignore it.
	// TODO: KindToResource only works for resources which can be derived from their kinds
	// by just converting the kind to lowercase and pluralizing it. This might not work for
	// all kinds and/or resources, example Node/Minion. A DiscoverRESTMapper should be able to
	// provide an appropriate mapping from kinds to resources and it is being worked on right
	// now. The only target resources for HPA Scale right now are replicationcontrollers,
	// replicasets and deployments, and all of them fall within the class of resources
	// that can be simply converted by pluralizing the lower case of their kind names. So
	// leaving KindToResource() function call for now. But it should be replaced with
	// DiscoveryRESTMapper when it is ready.
	gvr, _ := meta.KindToResource(gv.WithKind(scaleRef.Kind))

	resource := &unversioned.APIResource{
		Name:       gvr.Resource,
		Namespaced: true,
		Kind:       scaleRef.Kind,
	}
	res, err := c.client(gv).Resource(resource).Namespace(namespace).Subresource(scaleRef.Subresource).Get(scaleRef.Name)
	if err != nil {
		return nil, fmt.Errorf("unable to fetch resource %s(%s): %v", scaleRef.Name, scaleRef.Kind, err)
	}
	return c.decode(res)
}

func (c *Client) Update(scaleRef extensions.SubresourceReference, namespace string, obj ScalerInterface) (ScalerInterface, error) {
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
	res, err := c.client(gv).Resource(resource).Namespace(namespace).Subresource(scaleRef.Subresource).Update(obj.getScale())
	if err != nil {
		return nil, fmt.Errorf("unable to fetch resource %s(%s): %v", scaleRef.Name, scaleRef.Kind, err)
	}
	return c.decode(res)
}

func field(obj map[string]interface{}, keyChain ...string) (interface{}, error) {
	res := reflect.ValueOf(obj).Interface()
	for _, key := range keyChain {
		resVal, ok := res.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("%v should be of type map[string]interface{} to be able to proceed", res)
		}
		res, ok = resVal[key]
		if !ok {
			return nil, fmt.Errorf("key(`%s`) doesn't exist in the `Scale` object", key)
		}
	}
	return res, nil
}

type scale struct {
	c   *Client
	obj *runtime.Unstructured
}

func (s *scale) getScale() *runtime.Unstructured {
	return s.obj
}

func (s *scale) Replicas() (int, error) {
	ri, err := field(s.obj.Object, "status", "replicas")
	if err != nil {
		return 0, err
	}

	// Note: JSON only has a number type whose equivalent is float64 in Go. So we type-assert
	// to float64 here and then cast the value to int before returning.
	r, ok := ri.(float64)
	if !ok {
		return 0, fmt.Errorf("`Replicas` field isn't of expected type")
	}
	return int(r), nil
}

// TODO: Implementing this as a Get-Update cycle for now, but it should really be
// implemented as a Patch.
func (s *scale) SetReplicas(replicas int) {
	// Note: JSON only has a number type whose equivalent is float64 in Go. So we cast
	// replicas to float64 before updating.
	s.obj.Object["spec"] = map[string]float64{
		"replicas": float64(replicas),
	}
}

type extScale struct {
	scale
}

func (s *extScale) Selector() (labels.Selector, error) {
	seli, err := field(s.obj.Object, "status", "selector")
	if err != nil {
		return nil, err
	}
	selMap, ok := seli.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("`Selector` isn't of expected type")
	}

	selector := make(map[string]string)
	for key, val := range selMap {
		selector[key] = val.(string)
	}

	return labels.SelectorFromSet(selector), nil
}

type asScale struct {
	scale
}

func (s *asScale) Selector() (labels.Selector, error) {
	seli, err := field(s.obj.Object, "status", "selector")
	if err != nil {
		return nil, err
	}
	selector, ok := seli.(string)
	if !ok {
		return nil, fmt.Errorf("`Selector` isn't of expected type")
	}
	// TODO: Eliminate deserialization/reserialization of the selector.
	parsedSelector, err := labels.Parse(selector)
	if err != nil {
		return nil, fmt.Errorf("couldn't convert selector string to a corresponding selector object: %v", err)
	}
	return parsedSelector, nil
}
