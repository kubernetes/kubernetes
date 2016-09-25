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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/watch"
)

type PodSecurityPoliciesInterface interface {
	PodSecurityPolicies() PodSecurityPolicyInterface
}

type PodSecurityPolicyInterface interface {
	Get(name string) (result *extensions.PodSecurityPolicy, err error)
	Create(psp *extensions.PodSecurityPolicy) (*extensions.PodSecurityPolicy, error)
	List(opts api.ListOptions) (*extensions.PodSecurityPolicyList, error)
	Delete(name string) error
	Update(*extensions.PodSecurityPolicy) (*extensions.PodSecurityPolicy, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// podSecurityPolicy implements PodSecurityPolicyInterface
type podSecurityPolicy struct {
	client *ExtensionsClient
}

// newPodSecurityPolicy returns a podSecurityPolicy object.
func newPodSecurityPolicy(c *ExtensionsClient) *podSecurityPolicy {
	return &podSecurityPolicy{c}
}

func (s *podSecurityPolicy) Create(psp *extensions.PodSecurityPolicy) (*extensions.PodSecurityPolicy, error) {
	result := &extensions.PodSecurityPolicy{}
	err := s.client.Post().
		Resource("podsecuritypolicies").
		Body(psp).
		Do().
		Into(result)

	return result, err
}

// List returns a list of PodSecurityPolicies matching the selectors.
func (s *podSecurityPolicy) List(opts api.ListOptions) (*extensions.PodSecurityPolicyList, error) {
	result := &extensions.PodSecurityPolicyList{}

	err := s.client.Get().
		Resource("podsecuritypolicies").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)

	return result, err
}

// Get returns the given PodSecurityPolicy, or an error.
func (s *podSecurityPolicy) Get(name string) (*extensions.PodSecurityPolicy, error) {
	result := &extensions.PodSecurityPolicy{}
	err := s.client.Get().
		Resource("podsecuritypolicies").
		Name(name).
		Do().
		Into(result)

	return result, err
}

// Watch starts watching for PodSecurityPolicies matching the given selectors.
func (s *podSecurityPolicy) Watch(opts api.ListOptions) (watch.Interface, error) {
	return s.client.Get().
		Prefix("watch").
		Resource("podsecuritypolicies").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

func (s *podSecurityPolicy) Delete(name string) error {
	return s.client.Delete().
		Resource("podsecuritypolicies").
		Name(name).
		Do().
		Error()
}

func (s *podSecurityPolicy) Update(psp *extensions.PodSecurityPolicy) (result *extensions.PodSecurityPolicy, err error) {
	result = &extensions.PodSecurityPolicy{}
	err = s.client.Put().
		Resource("podsecuritypolicies").
		Name(psp.Name).
		Body(psp).
		Do().
		Into(result)

	return
}
