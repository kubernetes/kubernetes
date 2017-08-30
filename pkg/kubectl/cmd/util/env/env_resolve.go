/*
Copyright 2017 The Kubernetes Authors.

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

package env

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/fieldpath"
)

// ResourceStore defines a new resource store data structure
type ResourceStore struct {
	SecretStore    map[string]*api.Secret
	ConfigMapStore map[string]*api.ConfigMap
}

// NewResourceStore returns a pointer to a new resource store data structure
func NewResourceStore() *ResourceStore {
	return &ResourceStore{
		SecretStore:    make(map[string]*api.Secret),
		ConfigMapStore: make(map[string]*api.ConfigMap),
	}
}

// getSecretRefValue returns the value of a secret in the supplied namespace
func getSecretRefValue(client clientset.Interface, namespace string, store *ResourceStore, secretSelector *api.SecretKeySelector) (string, error) {
	secret, ok := store.SecretStore[secretSelector.Name]
	if !ok {
		var err error
		secret, err = client.Core().Secrets(namespace).Get(secretSelector.Name, metav1.GetOptions{})
		if err != nil {
			return "", err
		}
		store.SecretStore[secretSelector.Name] = secret
	}
	if data, ok := secret.Data[secretSelector.Key]; ok {
		return string(data), nil
	}
	return "", fmt.Errorf("key %s not found in secret %s", secretSelector.Key, secretSelector.Name)

}

// getConfigMapRefValue returns the value of a configmap in the supplied namespace
func getConfigMapRefValue(client clientset.Interface, namespace string, store *ResourceStore, configMapSelector *api.ConfigMapKeySelector) (string, error) {
	configMap, ok := store.ConfigMapStore[configMapSelector.Name]
	if !ok {
		var err error
		configMap, err = client.Core().ConfigMaps(namespace).Get(configMapSelector.Name, metav1.GetOptions{})
		if err != nil {
			return "", err
		}
		store.ConfigMapStore[configMapSelector.Name] = configMap
	}
	if data, ok := configMap.Data[configMapSelector.Key]; ok {
		return string(data), nil
	}
	return "", fmt.Errorf("key %s not found in config map %s", configMapSelector.Key, configMapSelector.Name)
}

// getFieldRef returns the value of the supplied path in the given object
func getFieldRef(obj runtime.Object, from *api.EnvVarSource) (string, error) {
	return fieldpath.ExtractFieldPathAsString(obj, from.FieldRef.FieldPath)
}

// getResourceFieldRef returns the value of a resource in the given container
func getResourceFieldRef(from *api.EnvVarSource, c *api.Container) (string, error) {
	return resource.ExtractContainerResourceValue(from.ResourceFieldRef, c)
}

// GetEnvVarRefValue returns the value referenced by the supplied envvarsource given the other supplied information
func GetEnvVarRefValue(kc clientset.Interface, ns string, store *ResourceStore, from *api.EnvVarSource, obj runtime.Object, c *api.Container) (string, error) {
	if from.SecretKeyRef != nil {
		return getSecretRefValue(kc, ns, store, from.SecretKeyRef)
	}

	if from.ConfigMapKeyRef != nil {
		return getConfigMapRefValue(kc, ns, store, from.ConfigMapKeyRef)
	}

	if from.FieldRef != nil {
		return getFieldRef(obj, from)
	}

	if from.ResourceFieldRef != nil {
		return getResourceFieldRef(from, c)
	}

	return "", fmt.Errorf("invalid valueFrom")
}

// GetEnvVarRefString returns a text description of the supplied envvarsource
func GetEnvVarRefString(from *api.EnvVarSource) string {
	if from.ConfigMapKeyRef != nil {
		return fmt.Sprintf("configmap %s, key %s", from.ConfigMapKeyRef.Name, from.ConfigMapKeyRef.Key)
	}

	if from.SecretKeyRef != nil {
		return fmt.Sprintf("secret %s, key %s", from.SecretKeyRef.Name, from.SecretKeyRef.Key)
	}

	if from.FieldRef != nil {
		return fmt.Sprintf("field path %s", from.FieldRef.FieldPath)
	}

	if from.ResourceFieldRef != nil {
		containerPrefix := ""
		if from.ResourceFieldRef.ContainerName != "" {
			containerPrefix = fmt.Sprintf("%s/", from.ResourceFieldRef.ContainerName)
		}
		return fmt.Sprintf("resource field %s%s", containerPrefix, from.ResourceFieldRef.Resource)
	}

	return "invalid valueFrom"
}
