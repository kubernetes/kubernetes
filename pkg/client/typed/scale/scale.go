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
	dc *dynamic.Client
}

type ScaleInterface interface {
	Replicas() (int, error)
	SetReplicas(replicas int)
	Selector() (string, error)
}

func NewClient(conf *restclient.Config) (*Client, error) {
	dc, err := dynamic.NewClient(conf)
	if err != nil {
		return nil, err
	}

	return &Client{
		dc: dc,
	}, nil
}

// NewForConfigOrDie creates a new dynamic client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(conf *restclient.Config) *Client {
	cl, err := NewClient(conf)
	if err != nil {
		panic(err)
	}
	return cl
}

// TODO(madhusudancs): Implement this as a codec decoder.
func (c *Client) decode(res *runtime.Unstructured) (ScaleInterface, error) {
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
	return c.decode(res)
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
		fmt.Printf("ri type: %T\n", ri)
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

func (s *extScale) Selector() (string, error) {
	seli, err := field(s.obj.Object, "status", "selector")
	if err != nil {
		return "", err
	}
	selMap, ok := seli.(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("`Selector` isn't of expected type")
	}

	selector := make(map[string]string)
	for key, val := range selMap {
		selector[key] = val.(string)
	}

	return labels.SelectorFromSet(selector).String(), nil
}

type asScale struct {
	scale
}

func (s *asScale) Selector() (string, error) {
	seli, err := field(s.obj.Object, "status", "selector")
	if err != nil {
		return "", err
	}
	selector, ok := seli.(string)
	if !ok {
		return "", fmt.Errorf("`Selector` isn't of expected type")
	}
	return selector, nil
}
