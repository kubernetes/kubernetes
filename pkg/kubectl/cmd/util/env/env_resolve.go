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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	"k8s.io/kubernetes/pkg/fieldpath"
)

// ResourceStore defines a new resource store data structure.
type ResourceStore struct {
	SecretStore    map[string]*v1.Secret
	ConfigMapStore map[string]*v1.ConfigMap
}

// NewResourceStore returns a pointer to a new resource store data structure.
func NewResourceStore() *ResourceStore {
	return &ResourceStore{
		SecretStore:    make(map[string]*v1.Secret),
		ConfigMapStore: make(map[string]*v1.ConfigMap),
	}
}

// getSecretRefValue returns the value of a secret in the supplied namespace
func getSecretRefValue(client kubernetes.Interface, namespace string, store *ResourceStore, secretSelector *v1.SecretKeySelector) (string, error) {
	secret, ok := store.SecretStore[secretSelector.Name]
	if !ok {
		var err error
		secret, err = client.CoreV1().Secrets(namespace).Get(secretSelector.Name, metav1.GetOptions{})
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
func getConfigMapRefValue(client kubernetes.Interface, namespace string, store *ResourceStore, configMapSelector *v1.ConfigMapKeySelector) (string, error) {
	configMap, ok := store.ConfigMapStore[configMapSelector.Name]
	if !ok {
		var err error
		configMap, err = client.CoreV1().ConfigMaps(namespace).Get(configMapSelector.Name, metav1.GetOptions{})
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
func getFieldRef(obj runtime.Object, from *v1.EnvVarSource) (string, error) {
	return fieldpath.ExtractFieldPathAsString(obj, from.FieldRef.FieldPath)
}

// getResourceFieldRef returns the value of a resource in the given container
func getResourceFieldRef(from *v1.EnvVarSource, c *v1.Container) (string, error) {
	return resource.ExtractContainerResourceValue(from.ResourceFieldRef, c)
}

// GetEnvVarRefValue returns the value referenced by the supplied EnvVarSource given the other supplied information.
func GetEnvVarRefValue(kc kubernetes.Interface, ns string, store *ResourceStore, from *v1.EnvVarSource, obj runtime.Object, c *v1.Container) (string, error) {
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

// GetEnvVarRefString returns a text description of whichever field is set within the supplied EnvVarSource argument.
func GetEnvVarRefString(from *v1.EnvVarSource) string {
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
