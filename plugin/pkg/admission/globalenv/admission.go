/*
Copyright 2022 The Kubernetes Authors.

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

// Package globalenv contains an admission controller that modifies every new Pod to force
// the env of all containers to add global envs. The global envs are defined in the namespace annotation.
// The annotation key is "global-envs". The annotation value is a string with `,` as split char.
// For example: "TEST=test,FOO=foo"
package globalenv

import (
	"context"
	"io"
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/pods"
)

var validEnvNameRegexp = regexp.MustCompile("[^a-zA-Z0-9_]")

const (
	// PluginName indicates name of admission plugin.
	PluginName = "GlobalEnv"
	// globalEnvKey is the annotation key that holds the global env of the namespace
	globalEnvKey = "global.env"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewGlobalEnv(), nil
	})
}

// GlobalEnv is an implementation of admission.Interface.
// It looks at namespace annotation for global env prefix.
// It will add extra env to all new pods according to the global env of the namespace.
type GlobalEnv struct {
	*admission.Handler
	client          kubernetes.Interface
	namespaceLister corev1listers.NamespaceLister
}

var _ admission.MutationInterface = &GlobalEnv{}
var _ admission.ValidationInterface = &GlobalEnv{}

// Admit makes an admission decision based on the request attributes
func (g *GlobalEnv) Admit(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	if shouldIgnore(attributes) {
		return nil
	}
	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	pods.VisitContainersWithPath(&pod.Spec, field.NewPath("spec"), func(c *api.Container, _ *field.Path) bool {
		globalEnv, err := g.getNamespaceGlobalEnv(pod.Namespace)
		if err != nil {
			klog.ErrorS(err, "Failed to get global env annotation of the namespace", "pod", klog.KObj(pod))
		}
		if len(globalEnv) > 0 {
			globalEnvKeyValue := make(map[string]string)
			for _, env := range globalEnv {
				globalEnvKeyValue[env.Name] = env.Value
			}
			var envs []api.EnvVar
			// if global env will override the pod env, log more information
			for _, env := range c.Env {
				if value, ok := globalEnvKeyValue[env.Name]; ok {
					envs = append(envs, api.EnvVar{Name: env.Name, Value: value})
					delete(globalEnvKeyValue, env.Name)
					if value != env.Value {
						klog.InfoS("Global Env will override pod env", "pod", klog.KObj(pod), "env", env)
					}
				} else {
					envs = append(envs, api.EnvVar{Name: env.Name, Value: env.Value})
				}
			}
			for key, value := range globalEnvKeyValue {
				envs = append(envs, api.EnvVar{Name: key, Value: value})
			}
			c.Env = envs
		}
		return true
	})

	return nil
}

// Validate makes sure that pod env is not overwrited by a different value
func (g *GlobalEnv) Validate(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if shouldIgnore(attributes) {
		return nil
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	var allErrs []error
	pods.VisitContainersWithPath(&pod.Spec, field.NewPath("spec"), func(c *api.Container, p *field.Path) bool {
		// TODO: Should return error if global env will override pod env?
		globalEnvKeyValue, err := g.getNamespaceGlobalEnvMap(pod.Namespace)
		if err != nil {
			klog.ErrorS(err, "Failed to get global env annotation of the namespace", "pod", klog.KObj(pod))
		}
		if len(globalEnvKeyValue) > 0 {
			for _, env := range c.Env {
				if value, ok := globalEnvKeyValue[env.Name]; ok && value != env.Value {
					allErrs = append(allErrs, field.Forbidden(p.Child("env"), "global env should not override current env"))
				}
			}
		}
		return true
	})
	if len(allErrs) > 0 {
		return utilerrors.NewAggregate(allErrs)
	}

	return nil
}

// SetExternalKubeClientSet sets th client
func (p *GlobalEnv) SetExternalKubeClientSet(client kubernetes.Interface) {
	p.client = client
}

// SetExternalKubeInformerFactory initializes the Informer Factory
func (p *GlobalEnv) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	p.namespaceLister = namespaceInformer.Lister()
	p.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

// in exceptional cases, this can result in two live calls, but once the cache catches up, that will stop.
func (g *GlobalEnv) getNamespace(nsName string) (*corev1.Namespace, error) {
	namespace, err := g.namespaceLister.Get(nsName)
	if apierrors.IsNotFound(err) {
		// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
		namespace, err = g.client.CoreV1().Namespaces().Get(context.TODO(), nsName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return nil, err
			}
			return nil, apierrors.NewInternalError(err)
		}
	} else if err != nil {
		return nil, apierrors.NewInternalError(err)
	}

	return namespace, nil
}
func (g *GlobalEnv) getNamespaceGlobalEnvMap(nsName string) (map[string]string, error) {
	ns, err := g.getNamespace(nsName)
	if err != nil {
		return nil, err
	}
	return extractGlobalEnvMap(ns)
}

func (g *GlobalEnv) getNamespaceGlobalEnv(nsName string) ([]api.EnvVar, error) {
	ns, err := g.getNamespace(nsName)
	if err != nil {
		return nil, err
	}
	return extractGlobalEnv(ns)
}

func extractGlobalEnvMap(ns *corev1.Namespace) (map[string]string, error) {
	globalEnvKeyValue := make(map[string]string)
	annotation := ns.Annotations["global.env"]
	envs := strings.Split(annotation, ",")
	for _, env := range envs {
		if len(env) == 0 {
			continue
		}
		envKey, envValue, _ := strings.Cut(env, "=")
		globalEnvKeyValue[keyToEnvName(envKey)] = envValue
	}
	return globalEnvKeyValue, nil
}

func extractGlobalEnv(ns *corev1.Namespace) ([]api.EnvVar, error) {
	var globalEnv []api.EnvVar
	annotation := ns.Annotations[globalEnvKey]
	envs := strings.Split(annotation, ",")
	for _, env := range envs {
		if len(env) == 0 {
			continue
		}
		envKey, envValue, _ := strings.Cut(env, "=")
		globalEnv = append(globalEnv, api.EnvVar{
			Name:  keyToEnvName(envKey),
			Value: envValue,
		})
	}
	return globalEnv, nil
}

// check if it's update and it doesn't change the env by the pod spec
func isUpdateWithNoEnv(attributes admission.Attributes) bool {
	if attributes.GetOperation() != admission.Update {
		return false
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		klog.Warningf("Resource was marked with kind Pod but pod was unable to be converted.")
		return false
	}

	oldPod, ok := attributes.GetOldObject().(*api.Pod)
	if !ok {
		klog.Warningf("Resource was marked with kind Pod but old pod was unable to be converted.")
		return false
	}

	oldEnv := []api.EnvVar{}
	pods.VisitContainersWithPath(&oldPod.Spec, field.NewPath("spec"), func(c *api.Container, _ *field.Path) bool {
		oldEnv = append(oldEnv, c.Env...)
		return true
	})

	newEnv := []api.EnvVar{}
	pods.VisitContainersWithPath(&pod.Spec, field.NewPath("spec"), func(c *api.Container, _ *field.Path) bool {
		newEnv = append(newEnv, c.Env...)
		return true
	})

	return envSliceEquals(oldEnv, newEnv)
}

// envSliceEquals will ignore order
func envSliceEquals(expected, current []api.EnvVar) bool {
	if len(expected) != len(current) {
		return false
	}
	diff := map[string]string{}
	for _, env := range expected {
		diff[env.Name] = env.Value
	}
	for _, env := range current {
		if diff[env.Name] != env.Value {
			return false
		} else {
			delete(diff, env.Name)
		}
	}
	return len(diff) == 0
}

func shouldIgnore(attributes admission.Attributes) bool {
	// Ignore all calls to subresources or resources other than pods.
	if len(attributes.GetSubresource()) != 0 || attributes.GetResource().GroupResource() != api.Resource("pods") {
		return true
	}

	if isUpdateWithNoEnv(attributes) {
		return true
	}
	return false
}

func keyToEnvName(key string) string {
	envName := strings.ToUpper(validEnvNameRegexp.ReplaceAllString(key, "_"))
	if envName != key {
		klog.InfoS("Key transerred to env name", "key", key, "envName", envName)
	}
	return envName
}

// NewGlobalEnv creates a new global env admission control handler
func NewGlobalEnv() *GlobalEnv {
	return &GlobalEnv{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}
