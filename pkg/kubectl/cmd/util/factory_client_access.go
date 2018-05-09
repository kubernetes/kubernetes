/*
Copyright 2016 The Kubernetes Authors.

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

// this file contains factories with no other dependencies

package util

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"k8s.io/api/core/v1"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	batchv2alpha1 "k8s.io/api/batch/v2alpha1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/resource"
)

type RESTClientGetter interface {
	ToRESTConfig() (*restclient.Config, error)
	ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error)
	ToRESTMapper() (meta.RESTMapper, error)
	ToRawKubeConfigLoader() clientcmd.ClientConfig
}

type ring0Factory struct {
	clientGetter RESTClientGetter
}

func NewClientAccessFactory(clientGetter RESTClientGetter) ClientAccessFactory {
	if clientGetter == nil {
		panic("attempt to instantiate client_access_factory with nil clientGetter")
	}

	f := &ring0Factory{
		clientGetter: clientGetter,
	}

	return f
}

func (f *ring0Factory) ClientConfig() (*restclient.Config, error) {
	return f.clientGetter.ToRESTConfig()
}

func (f *ring0Factory) RESTMapper() (meta.RESTMapper, error) {
	return f.clientGetter.ToRESTMapper()
}

func (f *ring0Factory) BareClientConfig() (*restclient.Config, error) {
	return f.clientGetter.ToRESTConfig()
}

func (f *ring0Factory) DiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	return f.clientGetter.ToDiscoveryClient()
}

func (f *ring0Factory) KubernetesClientSet() (*kubernetes.Clientset, error) {
	clientConfig, err := f.ClientConfig()
	if err != nil {
		return nil, err
	}
	return kubernetes.NewForConfig(clientConfig)
}

func (f *ring0Factory) ClientSet() (internalclientset.Interface, error) {
	clientConfig, err := f.ClientConfig()
	if err != nil {
		return nil, err
	}
	return internalclientset.NewForConfig(clientConfig)
}

func (f *ring0Factory) DynamicClient() (dynamic.Interface, error) {
	clientConfig, err := f.ClientConfig()
	if err != nil {
		return nil, err
	}
	return dynamic.NewForConfig(clientConfig)
}

// NewBuilder returns a new resource builder for structured api objects.
func (f *ring0Factory) NewBuilder() *resource.Builder {
	return resource.NewBuilder(f.clientGetter)
}

func (f *ring0Factory) RESTClient() (*restclient.RESTClient, error) {
	clientConfig, err := f.ClientConfig()
	if err != nil {
		return nil, err
	}
	setKubernetesDefaults(clientConfig)
	return restclient.RESTClientFor(clientConfig)
}

func (f *ring0Factory) UpdatePodSpecForObject(obj runtime.Object, fn func(*v1.PodSpec) error) (bool, error) {
	// TODO: replace with a swagger schema based approach (identify pod template via schema introspection)
	switch t := obj.(type) {
	case *v1.Pod:
		return true, fn(&t.Spec)
	// ReplicationController
	case *v1.ReplicationController:
		if t.Spec.Template == nil {
			t.Spec.Template = &v1.PodTemplateSpec{}
		}
		return true, fn(&t.Spec.Template.Spec)

	// Deployment
	case *extensionsv1beta1.Deployment:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta1.Deployment:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta2.Deployment:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1.Deployment:
		return true, fn(&t.Spec.Template.Spec)

	// DaemonSet
	case *extensionsv1beta1.DaemonSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta2.DaemonSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1.DaemonSet:
		return true, fn(&t.Spec.Template.Spec)

	// ReplicaSet
	case *extensionsv1beta1.ReplicaSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta2.ReplicaSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1.ReplicaSet:
		return true, fn(&t.Spec.Template.Spec)

	// StatefulSet
	case *appsv1beta1.StatefulSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta2.StatefulSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1.StatefulSet:
		return true, fn(&t.Spec.Template.Spec)

	// Job
	case *batchv1.Job:
		return true, fn(&t.Spec.Template.Spec)

	// CronJob
	case *batchv1beta1.CronJob:
		return true, fn(&t.Spec.JobTemplate.Spec.Template.Spec)
	case *batchv2alpha1.CronJob:
		return true, fn(&t.Spec.JobTemplate.Spec.Template.Spec)

	default:
		return false, fmt.Errorf("the object is not a pod or does not have a pod template")
	}
}

func (f *ring0Factory) MapBasedSelectorForObject(object runtime.Object) (string, error) {
	// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
	switch t := object.(type) {
	case *api.ReplicationController:
		return kubectl.MakeLabels(t.Spec.Selector), nil
	case *api.Pod:
		if len(t.Labels) == 0 {
			return "", fmt.Errorf("the pod has no labels and cannot be exposed")
		}
		return kubectl.MakeLabels(t.Labels), nil
	case *api.Service:
		if t.Spec.Selector == nil {
			return "", fmt.Errorf("the service has no pod selector set")
		}
		return kubectl.MakeLabels(t.Spec.Selector), nil
	case *extensions.Deployment:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil
	case *extensions.ReplicaSet:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil
	default:
		return "", fmt.Errorf("cannot extract pod selector from %T", object)
	}
}

func (f *ring0Factory) PortsForObject(object runtime.Object) ([]string, error) {
	// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
	switch t := object.(type) {
	case *api.ReplicationController:
		return getPorts(t.Spec.Template.Spec), nil
	case *api.Pod:
		return getPorts(t.Spec), nil
	case *api.Service:
		return getServicePorts(t.Spec), nil
	case *extensions.Deployment:
		return getPorts(t.Spec.Template.Spec), nil
	case *extensions.ReplicaSet:
		return getPorts(t.Spec.Template.Spec), nil
	default:
		return nil, fmt.Errorf("cannot extract ports from %T", object)
	}
}

func (f *ring0Factory) ProtocolsForObject(object runtime.Object) (map[string]string, error) {
	// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
	switch t := object.(type) {
	case *api.ReplicationController:
		return getProtocols(t.Spec.Template.Spec), nil
	case *api.Pod:
		return getProtocols(t.Spec), nil
	case *api.Service:
		return getServiceProtocols(t.Spec), nil
	case *extensions.Deployment:
		return getProtocols(t.Spec.Template.Spec), nil
	case *extensions.ReplicaSet:
		return getProtocols(t.Spec.Template.Spec), nil
	default:
		return nil, fmt.Errorf("cannot extract protocols from %T", object)
	}
}

func (f *ring0Factory) LabelsForObject(object runtime.Object) (map[string]string, error) {
	return meta.NewAccessor().Labels(object)
}

// Set showSecrets false to filter out stuff like secrets.
func (f *ring0Factory) Command(cmd *cobra.Command, showSecrets bool) string {
	if len(os.Args) == 0 {
		return ""
	}

	flags := ""
	parseFunc := func(flag *pflag.Flag, value string) error {
		flags = flags + " --" + flag.Name
		if set, ok := flag.Annotations["classified"]; showSecrets || !ok || len(set) == 0 {
			flags = flags + "=" + value
		} else {
			flags = flags + "=CLASSIFIED"
		}
		return nil
	}
	var err error
	err = cmd.Flags().ParseAll(os.Args[1:], parseFunc)
	if err != nil || !cmd.Flags().Parsed() {
		return ""
	}

	args := ""
	if arguments := cmd.Flags().Args(); len(arguments) > 0 {
		args = " " + strings.Join(arguments, " ")
	}

	base := filepath.Base(os.Args[0])
	return base + args + flags
}

func (f *ring0Factory) SuggestedPodTemplateResources() []schema.GroupResource {
	return []schema.GroupResource{
		{Resource: "replicationcontroller"},
		{Resource: "deployment"},
		{Resource: "daemonset"},
		{Resource: "job"},
		{Resource: "replicaset"},
	}
}

func (f *ring0Factory) Pauser(info *resource.Info) ([]byte, error) {
	switch obj := info.Object.(type) {
	case *extensions.Deployment:
		if obj.Spec.Paused {
			return nil, errors.New("is already paused")
		}
		obj.Spec.Paused = true
		return runtime.Encode(InternalVersionJSONEncoder(), info.Object)
	default:
		return nil, fmt.Errorf("pausing is not supported")
	}
}

func (f *ring0Factory) ResolveImage(name string) (string, error) {
	return name, nil
}

func (f *ring0Factory) Resumer(info *resource.Info) ([]byte, error) {
	switch obj := info.Object.(type) {
	case *extensions.Deployment:
		if !obj.Spec.Paused {
			return nil, errors.New("is not paused")
		}
		obj.Spec.Paused = false
		return runtime.Encode(InternalVersionJSONEncoder(), info.Object)
	default:
		return nil, fmt.Errorf("resuming is not supported")
	}
}

func (f *ring0Factory) DefaultNamespace() (string, bool, error) {
	return f.clientGetter.ToRawKubeConfigLoader().Namespace()
}

const (
	// TODO(sig-cli): Enforce consistent naming for generators here.
	// See discussion in https://github.com/kubernetes/kubernetes/issues/46237
	// before you add any more.
	RunV1GeneratorName                      = "run/v1"
	RunPodV1GeneratorName                   = "run-pod/v1"
	ServiceV1GeneratorName                  = "service/v1"
	ServiceV2GeneratorName                  = "service/v2"
	ServiceNodePortGeneratorV1Name          = "service-nodeport/v1"
	ServiceClusterIPGeneratorV1Name         = "service-clusterip/v1"
	ServiceLoadBalancerGeneratorV1Name      = "service-loadbalancer/v1"
	ServiceExternalNameGeneratorV1Name      = "service-externalname/v1"
	ServiceAccountV1GeneratorName           = "serviceaccount/v1"
	HorizontalPodAutoscalerV1GeneratorName  = "horizontalpodautoscaler/v1"
	DeploymentV1Beta1GeneratorName          = "deployment/v1beta1"
	DeploymentAppsV1Beta1GeneratorName      = "deployment/apps.v1beta1"
	DeploymentBasicV1Beta1GeneratorName     = "deployment-basic/v1beta1"
	DeploymentBasicAppsV1Beta1GeneratorName = "deployment-basic/apps.v1beta1"
	DeploymentBasicAppsV1GeneratorName      = "deployment-basic/apps.v1"
	JobV1GeneratorName                      = "job/v1"
	CronJobV2Alpha1GeneratorName            = "cronjob/v2alpha1"
	CronJobV1Beta1GeneratorName             = "cronjob/v1beta1"
	NamespaceV1GeneratorName                = "namespace/v1"
	ResourceQuotaV1GeneratorName            = "resourcequotas/v1"
	SecretV1GeneratorName                   = "secret/v1"
	SecretForDockerRegistryV1GeneratorName  = "secret-for-docker-registry/v1"
	SecretForTLSV1GeneratorName             = "secret-for-tls/v1"
	ConfigMapV1GeneratorName                = "configmap/v1"
	ClusterRoleBindingV1GeneratorName       = "clusterrolebinding.rbac.authorization.k8s.io/v1alpha1"
	RoleBindingV1GeneratorName              = "rolebinding.rbac.authorization.k8s.io/v1alpha1"
	ClusterV1Beta1GeneratorName             = "cluster/v1beta1"
	PodDisruptionBudgetV1GeneratorName      = "poddisruptionbudget/v1beta1"
	PodDisruptionBudgetV2GeneratorName      = "poddisruptionbudget/v1beta1/v2"
	PriorityClassV1Alpha1GeneratorName      = "priorityclass/v1alpha1"
)

// DefaultGenerators returns the set of default generators for use in Factory instances
func DefaultGenerators(cmdName string) map[string]kubectl.Generator {
	var generator map[string]kubectl.Generator
	switch cmdName {
	case "expose":
		generator = map[string]kubectl.Generator{
			ServiceV1GeneratorName: kubectl.ServiceGeneratorV1{},
			ServiceV2GeneratorName: kubectl.ServiceGeneratorV2{},
		}
	case "service-clusterip":
		generator = map[string]kubectl.Generator{
			ServiceClusterIPGeneratorV1Name: kubectl.ServiceClusterIPGeneratorV1{},
		}
	case "service-nodeport":
		generator = map[string]kubectl.Generator{
			ServiceNodePortGeneratorV1Name: kubectl.ServiceNodePortGeneratorV1{},
		}
	case "service-loadbalancer":
		generator = map[string]kubectl.Generator{
			ServiceLoadBalancerGeneratorV1Name: kubectl.ServiceLoadBalancerGeneratorV1{},
		}
	case "deployment":
		// Create Deployment has only StructuredGenerators and no
		// param-based Generators.
		// The StructuredGenerators are as follows (as of 2018-03-16):
		// DeploymentBasicV1Beta1GeneratorName -> kubectl.DeploymentBasicGeneratorV1
		// DeploymentBasicAppsV1Beta1GeneratorName -> kubectl.DeploymentBasicAppsGeneratorV1Beta1
		// DeploymentBasicAppsV1GeneratorName -> kubectl.DeploymentBasicAppsGeneratorV1
		generator = map[string]kubectl.Generator{}
	case "run":
		generator = map[string]kubectl.Generator{
			RunV1GeneratorName:                 kubectl.BasicReplicationController{},
			RunPodV1GeneratorName:              kubectl.BasicPod{},
			DeploymentV1Beta1GeneratorName:     kubectl.DeploymentV1Beta1{},
			DeploymentAppsV1Beta1GeneratorName: kubectl.DeploymentAppsV1Beta1{},
			JobV1GeneratorName:                 kubectl.JobV1{},
			CronJobV2Alpha1GeneratorName:       kubectl.CronJobV2Alpha1{},
			CronJobV1Beta1GeneratorName:        kubectl.CronJobV1Beta1{},
		}
	case "namespace":
		generator = map[string]kubectl.Generator{
			NamespaceV1GeneratorName: kubectl.NamespaceGeneratorV1{},
		}
	case "quota":
		generator = map[string]kubectl.Generator{
			ResourceQuotaV1GeneratorName: kubectl.ResourceQuotaGeneratorV1{},
		}
	case "secret":
		generator = map[string]kubectl.Generator{
			SecretV1GeneratorName: kubectl.SecretGeneratorV1{},
		}
	case "secret-for-docker-registry":
		generator = map[string]kubectl.Generator{
			SecretForDockerRegistryV1GeneratorName: kubectl.SecretForDockerRegistryGeneratorV1{},
		}
	case "secret-for-tls":
		generator = map[string]kubectl.Generator{
			SecretForTLSV1GeneratorName: kubectl.SecretForTLSGeneratorV1{},
		}
	}

	return generator
}

// fallbackGeneratorNameIfNecessary returns the name of the old generator
// if server does not support new generator. Otherwise, the
// generator string is returned unchanged.
//
// If the generator name is changed, print a warning message to let the user
// know.
func FallbackGeneratorNameIfNecessary(
	generatorName string,
	discoveryClient discovery.DiscoveryInterface,
	cmdErr io.Writer,
) (string, error) {
	switch generatorName {
	case DeploymentAppsV1Beta1GeneratorName:
		hasResource, err := HasResource(discoveryClient, appsv1beta1.SchemeGroupVersion.WithResource("deployments"))
		if err != nil {
			return "", err
		}
		if !hasResource {
			return FallbackGeneratorNameIfNecessary(DeploymentV1Beta1GeneratorName, discoveryClient, cmdErr)
		}
	case DeploymentV1Beta1GeneratorName:
		hasResource, err := HasResource(discoveryClient, extensionsv1beta1.SchemeGroupVersion.WithResource("deployments"))
		if err != nil {
			return "", err
		}
		if !hasResource {
			return RunV1GeneratorName, nil
		}
	case DeploymentBasicAppsV1GeneratorName:
		hasResource, err := HasResource(discoveryClient, appsv1.SchemeGroupVersion.WithResource("deployments"))
		if err != nil {
			return "", err
		}
		if !hasResource {
			return FallbackGeneratorNameIfNecessary(DeploymentBasicAppsV1Beta1GeneratorName, discoveryClient, cmdErr)
		}
	case DeploymentBasicAppsV1Beta1GeneratorName:
		hasResource, err := HasResource(discoveryClient, appsv1beta1.SchemeGroupVersion.WithResource("deployments"))
		if err != nil {
			return "", err
		}
		if !hasResource {
			return DeploymentBasicV1Beta1GeneratorName, nil
		}
	case JobV1GeneratorName:
		hasResource, err := HasResource(discoveryClient, batchv1.SchemeGroupVersion.WithResource("jobs"))
		if err != nil {
			return "", err
		}
		if !hasResource {
			return RunPodV1GeneratorName, nil
		}
	case CronJobV1Beta1GeneratorName:
		hasResource, err := HasResource(discoveryClient, batchv1beta1.SchemeGroupVersion.WithResource("cronjobs"))
		if err != nil {
			return "", err
		}
		if !hasResource {
			return FallbackGeneratorNameIfNecessary(CronJobV2Alpha1GeneratorName, discoveryClient, cmdErr)
		}
	case CronJobV2Alpha1GeneratorName:
		hasResource, err := HasResource(discoveryClient, batchv2alpha1.SchemeGroupVersion.WithResource("cronjobs"))
		if err != nil {
			return "", err
		}
		if !hasResource {
			return JobV1GeneratorName, nil
		}
	}
	return generatorName, nil
}

func Warning(cmdErr io.Writer, newGeneratorName, oldGeneratorName string) {
	fmt.Fprintf(cmdErr, "WARNING: New generator %q specified, "+
		"but it isn't available. "+
		"Falling back to %q.\n",
		newGeneratorName,
		oldGeneratorName,
	)
}

func HasResource(client discovery.DiscoveryInterface, resource schema.GroupVersionResource) (bool, error) {
	resources, err := client.ServerResourcesForGroupVersion(resource.GroupVersion().String())
	if apierrors.IsNotFound(err) {
		// entire group is missing
		return false, nil
	}
	if err != nil {
		// other errors error
		return false, fmt.Errorf("failed to discover supported resources: %v", err)
	}
	for _, serverResource := range resources.APIResources {
		if serverResource.Name == resource.Resource {
			return true, nil
		}
	}
	return false, nil
}

func Contains(resourcesList []*metav1.APIResourceList, resource schema.GroupVersionResource) bool {
	resources := discovery.FilteredBy(discovery.ResourcePredicateFunc(func(gv string, r *metav1.APIResource) bool {
		return resource.GroupVersion().String() == gv && resource.Resource == r.Name
	}), resourcesList)
	return len(resources) != 0
}

func (f *ring0Factory) Generators(cmdName string) map[string]kubectl.Generator {
	return DefaultGenerators(cmdName)
}

func (f *ring0Factory) CanBeExposed(kind schema.GroupKind) error {
	switch kind {
	case api.Kind("ReplicationController"), api.Kind("Service"), api.Kind("Pod"),
		extensions.Kind("Deployment"), apps.Kind("Deployment"), extensions.Kind("ReplicaSet"), apps.Kind("ReplicaSet"):
		// nothing to do here
	default:
		return fmt.Errorf("cannot expose a %s", kind)
	}
	return nil
}

func (f *ring0Factory) CanBeAutoscaled(kind schema.GroupKind) error {
	switch kind {
	case api.Kind("ReplicationController"), extensions.Kind("ReplicaSet"),
		extensions.Kind("Deployment"), apps.Kind("Deployment"), apps.Kind("ReplicaSet"):
		// nothing to do here
	default:
		return fmt.Errorf("cannot autoscale a %v", kind)
	}
	return nil
}

func (f *ring0Factory) EditorEnvs() []string {
	return []string{"KUBE_EDITOR", "EDITOR"}
}

// overlyCautiousIllegalFileCharacters matches characters that *might* not be supported.  Windows is really restrictive, so this is really restrictive
var overlyCautiousIllegalFileCharacters = regexp.MustCompile(`[^(\w/\.)]`)

// computeDiscoverCacheDir takes the parentDir and the host and comes up with a "usually non-colliding" name.
func computeDiscoverCacheDir(parentDir, host string) string {
	// strip the optional scheme from host if its there:
	schemelessHost := strings.Replace(strings.Replace(host, "https://", "", 1), "http://", "", 1)
	// now do a simple collapse of non-AZ09 characters.  Collisions are possible but unlikely.  Even if we do collide the problem is short lived
	safeHost := overlyCautiousIllegalFileCharacters.ReplaceAllString(schemelessHost, "_")

	return filepath.Join(parentDir, safeHost)
}

// this method exists to help us find the points still relying on internal types.
func InternalVersionDecoder() runtime.Decoder {
	return legacyscheme.Codecs.UniversalDecoder()
}

func InternalVersionJSONEncoder() runtime.Encoder {
	encoder := legacyscheme.Codecs.LegacyCodec(legacyscheme.Scheme.PrioritizedVersionsAllGroups()...)
	return unstructured.JSONFallbackEncoder{Encoder: encoder}
}
