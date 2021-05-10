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

package serviceaccount

import (
	"context"
	"fmt"
	"io"
	"math/rand"
	"strconv"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/utils/pointer"
)

const (
	// DefaultServiceAccountName is the name of the default service account to set on pods which do not specify a service account
	DefaultServiceAccountName = "default"

	// EnforceMountableSecretsAnnotation is a default annotation that indicates that a service account should enforce mountable secrets.
	// The value must be true to have this annotation take effect
	EnforceMountableSecretsAnnotation = "kubernetes.io/enforce-mountable-secrets"

	// ServiceAccountVolumeName is the prefix name that will be added to volumes that mount ServiceAccount secrets
	ServiceAccountVolumeName = "kube-api-access"

	// DefaultAPITokenMountPath is the path that ServiceAccountToken secrets are automounted to.
	// The token file would then be accessible at /var/run/secrets/kubernetes.io/serviceaccount
	DefaultAPITokenMountPath = "/var/run/secrets/kubernetes.io/serviceaccount"

	// PluginName is the name of this admission plugin
	PluginName = "ServiceAccount"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		serviceAccountAdmission := NewServiceAccount()
		return serviceAccountAdmission, nil
	})
}

var _ = admission.Interface(&Plugin{})

// Plugin contains the client used by the admission controller
type Plugin struct {
	*admission.Handler

	// LimitSecretReferences rejects pods that reference secrets their service accounts do not reference
	LimitSecretReferences bool
	// MountServiceAccountToken creates Volume and VolumeMounts for the first referenced ServiceAccountToken for the pod's service account
	MountServiceAccountToken bool

	client kubernetes.Interface

	serviceAccountLister corev1listers.ServiceAccountLister

	generateName func(string) string
}

var _ admission.MutationInterface = &Plugin{}
var _ admission.ValidationInterface = &Plugin{}
var _ = genericadmissioninitializer.WantsExternalKubeClientSet(&Plugin{})
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&Plugin{})

// NewServiceAccount returns an admission.Interface implementation which limits admission of Pod CREATE requests based on the pod's ServiceAccount:
// 1. If the pod does not specify a ServiceAccount, it sets the pod's ServiceAccount to "default"
// 2. It ensures the ServiceAccount referenced by the pod exists
// 3. If LimitSecretReferences is true, it rejects the pod if the pod references Secret objects which the pod's ServiceAccount does not reference
// 4. If the pod does not contain any ImagePullSecrets, the ImagePullSecrets of the service account are added.
// 5. If MountServiceAccountToken is true, it adds a VolumeMount with the pod's ServiceAccount's api token secret to containers
func NewServiceAccount() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
		// TODO: enable this once we've swept secret usage to account for adding secret references to service accounts
		LimitSecretReferences: false,
		// Auto mount service account API token secrets
		MountServiceAccountToken: true,

		generateName: names.SimpleNameGenerator.GenerateName,
	}
}

// SetExternalKubeClientSet sets the client for the plugin
func (s *Plugin) SetExternalKubeClientSet(cl kubernetes.Interface) {
	s.client = cl
}

// SetExternalKubeInformerFactory registers informers with the plugin
func (s *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	serviceAccountInformer := f.Core().V1().ServiceAccounts()
	s.serviceAccountLister = serviceAccountInformer.Lister()
	s.SetReadyFunc(func() bool {
		return serviceAccountInformer.Informer().HasSynced()
	})
}

// ValidateInitialization ensures an authorizer is set.
func (s *Plugin) ValidateInitialization() error {
	if s.client == nil {
		return fmt.Errorf("missing client")
	}
	if s.serviceAccountLister == nil {
		return fmt.Errorf("missing serviceAccountLister")
	}
	return nil
}

// Admit verifies if the pod should be admitted
func (s *Plugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if shouldIgnore(a) {
		return nil
	}
	if a.GetOperation() != admission.Create {
		// we only mutate pods during create requests
		return nil
	}
	pod := a.GetObject().(*api.Pod)

	// Don't modify the spec of mirror pods.
	// That makes the kubelet very angry and confused, and it immediately deletes the pod (because the spec doesn't match)
	// That said, don't allow mirror pods to reference ServiceAccounts or SecretVolumeSources either
	if _, isMirrorPod := pod.Annotations[api.MirrorPodAnnotationKey]; isMirrorPod {
		return s.Validate(ctx, a, o)
	}

	// Set the default service account if needed
	if len(pod.Spec.ServiceAccountName) == 0 {
		pod.Spec.ServiceAccountName = DefaultServiceAccountName
	}

	serviceAccount, err := s.getServiceAccount(a.GetNamespace(), pod.Spec.ServiceAccountName)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("error looking up service account %s/%s: %w", a.GetNamespace(), pod.Spec.ServiceAccountName, err))
	}
	if s.MountServiceAccountToken && shouldAutomount(serviceAccount, pod) {
		s.mountServiceAccountToken(serviceAccount, pod)
	}
	if len(pod.Spec.ImagePullSecrets) == 0 {
		pod.Spec.ImagePullSecrets = make([]api.LocalObjectReference, len(serviceAccount.ImagePullSecrets))
		for i := 0; i < len(serviceAccount.ImagePullSecrets); i++ {
			pod.Spec.ImagePullSecrets[i].Name = serviceAccount.ImagePullSecrets[i].Name
		}
	}

	return s.Validate(ctx, a, o)
}

// Validate the data we obtained
func (s *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if shouldIgnore(a) {
		return nil
	}

	pod := a.GetObject().(*api.Pod)

	if a.GetOperation() == admission.Update && a.GetSubresource() == "ephemeralcontainers" {
		return s.limitEphemeralContainerSecretReferences(pod, a)
	}

	if a.GetOperation() != admission.Create {
		// we only validate pod specs during create requests
		return nil
	}

	// Mirror pods have restrictions on what they can reference
	if _, isMirrorPod := pod.Annotations[api.MirrorPodAnnotationKey]; isMirrorPod {
		if len(pod.Spec.ServiceAccountName) != 0 {
			return admission.NewForbidden(a, fmt.Errorf("a mirror pod may not reference service accounts"))
		}
		hasSecrets := false
		podutil.VisitPodSecretNames(pod, func(name string) bool {
			hasSecrets = true
			return false
		}, podutil.AllContainers)
		if hasSecrets {
			return admission.NewForbidden(a, fmt.Errorf("a mirror pod may not reference secrets"))
		}
		for _, v := range pod.Spec.Volumes {
			if proj := v.Projected; proj != nil {
				for _, projSource := range proj.Sources {
					if projSource.ServiceAccountToken != nil {
						return admission.NewForbidden(a, fmt.Errorf("a mirror pod may not use ServiceAccountToken volume projections"))
					}
				}
			}
		}
		return nil
	}

	// Require container pods to have service accounts
	if len(pod.Spec.ServiceAccountName) == 0 {
		return admission.NewForbidden(a, fmt.Errorf("no service account specified for pod %s/%s", a.GetNamespace(), pod.Name))
	}
	// Ensure the referenced service account exists
	serviceAccount, err := s.getServiceAccount(a.GetNamespace(), pod.Spec.ServiceAccountName)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("error looking up service account %s/%s: %v", a.GetNamespace(), pod.Spec.ServiceAccountName, err))
	}

	if s.enforceMountableSecrets(serviceAccount) {
		if err := s.limitSecretReferences(serviceAccount, pod); err != nil {
			return admission.NewForbidden(a, err)
		}
	}

	return nil
}

func shouldIgnore(a admission.Attributes) bool {
	if a.GetResource().GroupResource() != api.Resource("pods") || (a.GetSubresource() != "" && a.GetSubresource() != "ephemeralcontainers") {
		return true
	}
	obj := a.GetObject()
	if obj == nil {
		return true
	}
	_, ok := obj.(*api.Pod)
	if !ok {
		return true
	}

	return false
}

func shouldAutomount(sa *corev1.ServiceAccount, pod *api.Pod) bool {
	// Pod's preference wins
	if pod.Spec.AutomountServiceAccountToken != nil {
		return *pod.Spec.AutomountServiceAccountToken
	}
	// Then service account's
	if sa.AutomountServiceAccountToken != nil {
		return *sa.AutomountServiceAccountToken
	}
	// Default to true for backwards compatibility
	return true
}

// enforceMountableSecrets indicates whether mountable secrets should be enforced for a particular service account
// A global setting of true will override any flag set on the individual service account
func (s *Plugin) enforceMountableSecrets(serviceAccount *corev1.ServiceAccount) bool {
	if s.LimitSecretReferences {
		return true
	}

	if value, ok := serviceAccount.Annotations[EnforceMountableSecretsAnnotation]; ok {
		enforceMountableSecretCheck, _ := strconv.ParseBool(value)
		return enforceMountableSecretCheck
	}

	return false
}

// getServiceAccount returns the ServiceAccount for the given namespace and name if it exists
func (s *Plugin) getServiceAccount(namespace string, name string) (*corev1.ServiceAccount, error) {
	serviceAccount, err := s.serviceAccountLister.ServiceAccounts(namespace).Get(name)
	if err == nil {
		return serviceAccount, nil
	}
	if !errors.IsNotFound(err) {
		return nil, err
	}

	// Could not find in cache, attempt to look up directly
	numAttempts := 1
	if name == DefaultServiceAccountName {
		// If this is the default serviceaccount, attempt more times, since it should be auto-created by the controller
		numAttempts = 10
	}
	retryInterval := time.Duration(rand.Int63n(100)+int64(100)) * time.Millisecond
	for i := 0; i < numAttempts; i++ {
		if i != 0 {
			time.Sleep(retryInterval)
		}
		serviceAccount, err := s.client.CoreV1().ServiceAccounts(namespace).Get(context.TODO(), name, metav1.GetOptions{})
		if err == nil {
			return serviceAccount, nil
		}
		if !errors.IsNotFound(err) {
			return nil, err
		}
	}

	return nil, errors.NewNotFound(api.Resource("serviceaccount"), name)
}

func (s *Plugin) limitSecretReferences(serviceAccount *corev1.ServiceAccount, pod *api.Pod) error {
	// Ensure all secrets the pod references are allowed by the service account
	mountableSecrets := sets.NewString()
	for _, s := range serviceAccount.Secrets {
		mountableSecrets.Insert(s.Name)
	}
	for _, volume := range pod.Spec.Volumes {
		source := volume.VolumeSource
		if source.Secret == nil {
			continue
		}
		secretName := source.Secret.SecretName
		if !mountableSecrets.Has(secretName) {
			return fmt.Errorf("volume with secret.secretName=\"%s\" is not allowed because service account %s does not reference that secret", secretName, serviceAccount.Name)
		}
	}

	for _, container := range pod.Spec.InitContainers {
		for _, env := range container.Env {
			if env.ValueFrom != nil && env.ValueFrom.SecretKeyRef != nil {
				if !mountableSecrets.Has(env.ValueFrom.SecretKeyRef.Name) {
					return fmt.Errorf("init container %s with envVar %s referencing secret.secretName=\"%s\" is not allowed because service account %s does not reference that secret", container.Name, env.Name, env.ValueFrom.SecretKeyRef.Name, serviceAccount.Name)
				}
			}
		}
		for _, envFrom := range container.EnvFrom {
			if envFrom.SecretRef != nil {
				if !mountableSecrets.Has(envFrom.SecretRef.Name) {
					return fmt.Errorf("init container %s with envFrom referencing secret.secretName=\"%s\" is not allowed because service account %s does not reference that secret", container.Name, envFrom.SecretRef.Name, serviceAccount.Name)
				}
			}
		}
	}

	for _, container := range pod.Spec.Containers {
		for _, env := range container.Env {
			if env.ValueFrom != nil && env.ValueFrom.SecretKeyRef != nil {
				if !mountableSecrets.Has(env.ValueFrom.SecretKeyRef.Name) {
					return fmt.Errorf("container %s with envVar %s referencing secret.secretName=\"%s\" is not allowed because service account %s does not reference that secret", container.Name, env.Name, env.ValueFrom.SecretKeyRef.Name, serviceAccount.Name)
				}
			}
		}
		for _, envFrom := range container.EnvFrom {
			if envFrom.SecretRef != nil {
				if !mountableSecrets.Has(envFrom.SecretRef.Name) {
					return fmt.Errorf("container %s with envFrom referencing secret.secretName=\"%s\" is not allowed because service account %s does not reference that secret", container.Name, envFrom.SecretRef.Name, serviceAccount.Name)
				}
			}
		}
	}

	// limit pull secret references as well
	pullSecrets := sets.NewString()
	for _, s := range serviceAccount.ImagePullSecrets {
		pullSecrets.Insert(s.Name)
	}
	for i, pullSecretRef := range pod.Spec.ImagePullSecrets {
		if !pullSecrets.Has(pullSecretRef.Name) {
			return fmt.Errorf(`imagePullSecrets[%d].name="%s" is not allowed because service account %s does not reference that imagePullSecret`, i, pullSecretRef.Name, serviceAccount.Name)
		}
	}
	return nil
}

func (s *Plugin) limitEphemeralContainerSecretReferences(pod *api.Pod, a admission.Attributes) error {
	// Require ephemeral container pods to have service accounts
	if len(pod.Spec.ServiceAccountName) == 0 {
		return admission.NewForbidden(a, fmt.Errorf("no service account specified for pod %s/%s", a.GetNamespace(), pod.Name))
	}
	// Ensure the referenced service account exists
	serviceAccount, err := s.getServiceAccount(a.GetNamespace(), pod.Spec.ServiceAccountName)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("error looking up service account %s/%s: %w", a.GetNamespace(), pod.Spec.ServiceAccountName, err))
	}
	if !s.enforceMountableSecrets(serviceAccount) {
		return nil
	}
	// Ensure all secrets the ephemeral containers reference are allowed by the service account
	mountableSecrets := sets.NewString()
	for _, s := range serviceAccount.Secrets {
		mountableSecrets.Insert(s.Name)
	}
	for _, container := range pod.Spec.EphemeralContainers {
		for _, env := range container.Env {
			if env.ValueFrom != nil && env.ValueFrom.SecretKeyRef != nil {
				if !mountableSecrets.Has(env.ValueFrom.SecretKeyRef.Name) {
					return fmt.Errorf("ephemeral container %s with envVar %s referencing secret.secretName=\"%s\" is not allowed because service account %s does not reference that secret", container.Name, env.Name, env.ValueFrom.SecretKeyRef.Name, serviceAccount.Name)
				}
			}
		}
		for _, envFrom := range container.EnvFrom {
			if envFrom.SecretRef != nil {
				if !mountableSecrets.Has(envFrom.SecretRef.Name) {
					return fmt.Errorf("ephemeral container %s with envFrom referencing secret.secretName=\"%s\" is not allowed because service account %s does not reference that secret", container.Name, envFrom.SecretRef.Name, serviceAccount.Name)
				}
			}
		}
	}
	return nil
}

func (s *Plugin) mountServiceAccountToken(serviceAccount *corev1.ServiceAccount, pod *api.Pod) {
	// Find the volume and volume name for the ServiceAccountTokenSecret if it already exists
	tokenVolumeName := ""
	hasTokenVolume := false
	allVolumeNames := sets.NewString()
	for _, volume := range pod.Spec.Volumes {
		allVolumeNames.Insert(volume.Name)
		if strings.HasPrefix(volume.Name, ServiceAccountVolumeName+"-") {
			tokenVolumeName = volume.Name
			hasTokenVolume = true
			break
		}
	}

	// Determine a volume name for the ServiceAccountTokenSecret in case we need it
	if len(tokenVolumeName) == 0 {
		tokenVolumeName = s.generateName(ServiceAccountVolumeName + "-")
	}

	// Create the prototypical VolumeMount
	volumeMount := api.VolumeMount{
		Name:      tokenVolumeName,
		ReadOnly:  true,
		MountPath: DefaultAPITokenMountPath,
	}

	// Ensure every container mounts the APISecret volume
	needsTokenVolume := false
	for i, container := range pod.Spec.InitContainers {
		existingContainerMount := false
		for _, volumeMount := range container.VolumeMounts {
			// Existing mounts at the default mount path prevent mounting of the API token
			if volumeMount.MountPath == DefaultAPITokenMountPath {
				existingContainerMount = true
				break
			}
		}
		if !existingContainerMount {
			pod.Spec.InitContainers[i].VolumeMounts = append(pod.Spec.InitContainers[i].VolumeMounts, volumeMount)
			needsTokenVolume = true
		}
	}
	for i, container := range pod.Spec.Containers {
		existingContainerMount := false
		for _, volumeMount := range container.VolumeMounts {
			// Existing mounts at the default mount path prevent mounting of the API token
			if volumeMount.MountPath == DefaultAPITokenMountPath {
				existingContainerMount = true
				break
			}
		}
		if !existingContainerMount {
			pod.Spec.Containers[i].VolumeMounts = append(pod.Spec.Containers[i].VolumeMounts, volumeMount)
			needsTokenVolume = true
		}
	}

	// Add the volume if a container needs it
	if !hasTokenVolume && needsTokenVolume {
		pod.Spec.Volumes = append(pod.Spec.Volumes, api.Volume{
			Name: tokenVolumeName,
			VolumeSource: api.VolumeSource{
				Projected: TokenVolumeSource(),
			},
		})
	}
}

// TokenVolumeSource returns the projected volume source for service account token.
func TokenVolumeSource() *api.ProjectedVolumeSource {
	return &api.ProjectedVolumeSource{
		// explicitly set default value, see #104464
		DefaultMode: pointer.Int32(corev1.ProjectedVolumeSourceDefaultMode),
		Sources: []api.VolumeProjection{
			{
				ServiceAccountToken: &api.ServiceAccountTokenProjection{
					Path:              "token",
					ExpirationSeconds: serviceaccount.WarnOnlyBoundTokenExpirationSeconds,
				},
			},
			{
				ConfigMap: &api.ConfigMapProjection{
					LocalObjectReference: api.LocalObjectReference{
						Name: "kube-root-ca.crt",
					},
					Items: []api.KeyToPath{
						{
							Key:  "ca.crt",
							Path: "ca.crt",
						},
					},
				},
			},
			{
				DownwardAPI: &api.DownwardAPIProjection{
					Items: []api.DownwardAPIVolumeFile{
						{
							Path: "namespace",
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.namespace",
							},
						},
					},
				},
			},
			{
				ConfigMap: &api.ConfigMapProjection{
					LocalObjectReference: api.LocalObjectReference{
						Name: "openshift-service-ca.crt",
					},
					Items: []api.KeyToPath{
						{
							Key:  "service-ca.crt",
							Path: "service-ca.crt",
						},
					},
				},
			},
		},
	}
}
