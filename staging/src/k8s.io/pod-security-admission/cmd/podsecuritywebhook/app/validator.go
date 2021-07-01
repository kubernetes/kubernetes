/*
Copyright 2020 The Kubernetes Authors.

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

package app

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	podsecurityadmission "k8s.io/pod-security-admission/admission"
	podsecurityconfigloader "k8s.io/pod-security-admission/admission/api/load"
	"k8s.io/pod-security-admission/policy"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

type Validator struct {
	decoder  *admission.Decoder
	delegate *podsecurityadmission.Admission
}

func NewValidator(client client.Client, cache cache.Cache, configFile string) (*Validator, error) {
	_, err := cache.GetInformer(context.Background(), &corev1.Namespace{}) // Ensure that namespaces are cached.
	if err != nil {
		return nil, err
	}
	namespaceGetter := &namespaceGetter{client}
	podLister := &podLister{client}

	evaluator, err := policy.NewEvaluator(policy.DefaultChecks())
	if err != nil {
		return nil, fmt.Errorf("could not create PodSecurityRegistry: %w", err)
	}

	config, err := podsecurityconfigloader.LoadFromFile(configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load config file: %w", err)
	}

	delegate := &podsecurityadmission.Admission{
		Configuration:    config,
		Evaluator:        evaluator,
		Metrics:          nil, // TODO: wire to default prometheus metrics
		PodSpecExtractor: podsecurityadmission.DefaultPodSpecExtractor{},
		PodLister:        podLister,
		NamespaceGetter:  namespaceGetter,
	}

	if err := delegate.CompleteConfiguration(); err != nil {
		return nil, fmt.Errorf("configuration error: %w", err)
	}
	if err := delegate.ValidateConfiguration(); err != nil {
		return nil, fmt.Errorf("invalid: %w", err)
	}

	return &Validator{
		delegate: delegate,
	}, nil
}

// Handle handles the /validate-service endpoint requests
func (v *Validator) Handle(ctx context.Context, req admission.Request) admission.Response {
	attrs := podsecurityadmission.RequestAttributes(req.AdmissionRequest, v.decoder)
	resp := v.delegate.Validate(ctx, attrs)
	return admission.Response{AdmissionResponse: resp}
}

// InjectDecoder injects decoder into ServiceValidator
func (sv *Validator) InjectDecoder(d *admission.Decoder) error {
	sv.decoder = d
	return nil
}

type podLister struct {
	client.Reader
}

func (p *podLister) ListPods(ctx context.Context, namespace string) ([]*corev1.Pod, error) {
	list := &corev1.PodList{}
	err := p.Reader.List(ctx, list, &client.ListOptions{Namespace: namespace})
	if err != nil {
		return nil, err
	}
	pods := make([]*corev1.Pod, len(list.Items))
	for i := range list.Items {
		pods[i] = &list.Items[i]
	}
	return pods, nil
}

type namespaceGetter struct {
	client.Reader
}

func (n *namespaceGetter) GetNamespace(ctx context.Context, name string) (*corev1.Namespace, error) {
	namespace := &corev1.Namespace{}
	if err := n.Reader.Get(ctx, client.ObjectKey{Name: name}, namespace); err != nil {
		return nil, err
	}
	return namespace, nil
}
