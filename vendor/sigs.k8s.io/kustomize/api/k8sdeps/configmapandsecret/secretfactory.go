// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package configmapandsecret

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/kustomize/api/types"
)

func makeFreshSecret(
	args *types.SecretArgs) *corev1.Secret {
	s := &corev1.Secret{}
	s.APIVersion = "v1"
	s.Kind = "Secret"
	s.Name = args.Name
	s.Namespace = args.Namespace
	s.Type = corev1.SecretType(args.Type)
	if s.Type == "" {
		s.Type = corev1.SecretTypeOpaque
	}
	s.Data = map[string][]byte{}
	return s
}

// MakeSecret returns a new secret.
func (f *Factory) MakeSecret(
	args *types.SecretArgs) (*corev1.Secret, error) {
	all, err := f.kvLdr.Load(args.KvPairSources)
	if err != nil {
		return nil, err
	}
	s := makeFreshSecret(args)
	for _, p := range all {
		err = f.addKvToSecret(s, p.Key, p.Value)
		if err != nil {
			return nil, err
		}
	}
	if f.options != nil {
		s.SetLabels(f.options.Labels)
		s.SetAnnotations(f.options.Annotations)
	}
	return s, nil
}

func (f *Factory) addKvToSecret(secret *corev1.Secret, keyName, data string) error {
	if err := f.kvLdr.Validator().ErrIfInvalidKey(keyName); err != nil {
		return err
	}
	if _, entryExists := secret.Data[keyName]; entryExists {
		return fmt.Errorf(keyExistsErrorMsg, keyName, secret.Data)
	}
	secret.Data[keyName] = []byte(data)
	return nil
}
