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

package podpreset

import (
	"fmt"
	"io"
	"reflect"
	"strings"

	"k8s.io/klog"

	settingsv1alpha1 "k8s.io/api/settings/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	settingsv1alpha1listers "k8s.io/client-go/listers/settings/v1alpha1"
	api "k8s.io/kubernetes/pkg/apis/core"
	apiscorev1 "k8s.io/kubernetes/pkg/apis/core/v1"
)

const (
	annotationPrefix = "podpreset.admission.kubernetes.io"
	PluginName       = "PodPreset"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// podPresetPlugin is an implementation of admission.Interface.
type podPresetPlugin struct {
	*admission.Handler
	client kubernetes.Interface

	lister settingsv1alpha1listers.PodPresetLister
}

var _ admission.MutationInterface = &podPresetPlugin{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&podPresetPlugin{})
var _ = genericadmissioninitializer.WantsExternalKubeClientSet(&podPresetPlugin{})

// NewPlugin creates a new pod preset admission plugin.
func NewPlugin() *podPresetPlugin {
	return &podPresetPlugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

func (plugin *podPresetPlugin) ValidateInitialization() error {
	if plugin.client == nil {
		return fmt.Errorf("%s requires a client", PluginName)
	}
	if plugin.lister == nil {
		return fmt.Errorf("%s requires a lister", PluginName)
	}
	return nil
}

func (a *podPresetPlugin) SetExternalKubeClientSet(client kubernetes.Interface) {
	a.client = client
}

func (a *podPresetPlugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	podPresetInformer := f.Settings().V1alpha1().PodPresets()
	a.lister = podPresetInformer.Lister()
	a.SetReadyFunc(podPresetInformer.Informer().HasSynced)
}

// Admit injects a pod with the specific fields for each pod preset it matches.
func (c *podPresetPlugin) Admit(a admission.Attributes, o admission.ObjectInterfaces) error {
	// Ignore all calls to subresources or resources other than pods.
	// Ignore all operations other than CREATE.
	if len(a.GetSubresource()) != 0 || a.GetResource().GroupResource() != api.Resource("pods") || a.GetOperation() != admission.Create {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return errors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	if _, isMirrorPod := pod.Annotations[api.MirrorPodAnnotationKey]; isMirrorPod {
		return nil
	}

	// Ignore if exclusion annotation is present
	if podAnnotations := pod.GetAnnotations(); podAnnotations != nil {
		klog.V(5).Infof("Looking at pod annotations, found: %v", podAnnotations)
		if podAnnotations[api.PodPresetOptOutAnnotationKey] == "true" {
			return nil
		}
	}

	list, err := c.lister.PodPresets(a.GetNamespace()).List(labels.Everything())
	if err != nil {
		return fmt.Errorf("listing pod presets failed: %v", err)
	}

	matchingPPs, err := filterPodPresets(list, pod)
	if err != nil {
		return fmt.Errorf("filtering pod presets failed: %v", err)
	}

	if len(matchingPPs) == 0 {
		return nil
	}

	presetNames := make([]string, len(matchingPPs))
	for i, pp := range matchingPPs {
		presetNames[i] = pp.GetName()
	}

	// detect merge conflict
	err = safeToApplyPodPresetsOnPod(pod, matchingPPs)
	if err != nil {
		// conflict, ignore the error, but raise an event
		klog.Warningf("conflict occurred while applying podpresets: %s on pod: %v err: %v",
			strings.Join(presetNames, ","), pod.GetGenerateName(), err)
		return nil
	}

	applyPodPresetsOnPod(pod, matchingPPs)

	klog.Infof("applied podpresets: %s successfully on Pod: %+v ", strings.Join(presetNames, ","), pod.GetGenerateName())

	return nil
}

// filterPodPresets returns list of PodPresets which match given Pod.
func filterPodPresets(list []*settingsv1alpha1.PodPreset, pod *api.Pod) ([]*settingsv1alpha1.PodPreset, error) {
	var matchingPPs []*settingsv1alpha1.PodPreset

	for _, pp := range list {
		selector, err := metav1.LabelSelectorAsSelector(&pp.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("label selector conversion failed: %v for selector: %v", pp.Spec.Selector, err)
		}

		// check if the pod labels match the selector
		if !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		klog.V(4).Infof("PodPreset %s matches pod %s labels", pp.GetName(), pod.GetName())
		matchingPPs = append(matchingPPs, pp)
	}
	return matchingPPs, nil
}

// safeToApplyPodPresetsOnPod determines if there is any conflict in information
// injected by given PodPresets in the Pod.
func safeToApplyPodPresetsOnPod(pod *api.Pod, podPresets []*settingsv1alpha1.PodPreset) error {
	var errs []error

	// volumes attribute is defined at the Pod level, so determine if volumes
	// injection is causing any conflict.
	if _, err := mergeVolumes(pod.Spec.Volumes, podPresets); err != nil {
		errs = append(errs, err)
	}
	for _, ctr := range pod.Spec.Containers {
		if err := safeToApplyPodPresetsOnContainer(&ctr, podPresets); err != nil {
			errs = append(errs, err)
		}
	}
	for _, iCtr := range pod.Spec.InitContainers {
		if err := safeToApplyPodPresetsOnContainer(&iCtr, podPresets); err != nil {
			errs = append(errs, err)
		}
	}

	return utilerrors.NewAggregate(errs)
}

// safeToApplyPodPresetsOnContainer determines if there is any conflict in
// information injected by given PodPresets in the given container.
func safeToApplyPodPresetsOnContainer(ctr *api.Container, podPresets []*settingsv1alpha1.PodPreset) error {
	var errs []error
	// check if it is safe to merge env vars and volume mounts from given podpresets and
	// container's existing env vars.
	if _, err := mergeEnv(ctr.Env, podPresets); err != nil {
		errs = append(errs, err)
	}
	if _, err := mergeVolumeMounts(ctr.VolumeMounts, podPresets); err != nil {
		errs = append(errs, err)
	}

	return utilerrors.NewAggregate(errs)
}

// mergeEnv merges a list of env vars with the env vars injected by given list podPresets.
// It returns an error if it detects any conflict during the merge.
func mergeEnv(envVars []api.EnvVar, podPresets []*settingsv1alpha1.PodPreset) ([]api.EnvVar, error) {
	origEnv := map[string]api.EnvVar{}
	for _, v := range envVars {
		origEnv[v.Name] = v
	}

	mergedEnv := make([]api.EnvVar, len(envVars))
	copy(mergedEnv, envVars)

	var errs []error

	for _, pp := range podPresets {
		for _, v := range pp.Spec.Env {
			internalEnv := api.EnvVar{}
			if err := apiscorev1.Convert_v1_EnvVar_To_core_EnvVar(&v, &internalEnv, nil); err != nil {
				return nil, err
			}

			found, ok := origEnv[v.Name]
			if !ok {
				// if we don't already have it append it and continue
				origEnv[v.Name] = internalEnv
				mergedEnv = append(mergedEnv, internalEnv)
				continue
			}

			// make sure they are identical or throw an error
			if !reflect.DeepEqual(found, internalEnv) {
				errs = append(errs, fmt.Errorf("merging env for %s has a conflict on %s: \n%#v\ndoes not match\n%#v\n in container", pp.GetName(), v.Name, v, found))
			}
		}
	}

	err := utilerrors.NewAggregate(errs)
	if err != nil {
		return nil, err
	}

	return mergedEnv, err
}

func mergeEnvFrom(envSources []api.EnvFromSource, podPresets []*settingsv1alpha1.PodPreset) ([]api.EnvFromSource, error) {
	var mergedEnvFrom []api.EnvFromSource

	mergedEnvFrom = append(mergedEnvFrom, envSources...)
	for _, pp := range podPresets {
		for _, envFromSource := range pp.Spec.EnvFrom {
			internalEnvFrom := api.EnvFromSource{}
			if err := apiscorev1.Convert_v1_EnvFromSource_To_core_EnvFromSource(&envFromSource, &internalEnvFrom, nil); err != nil {
				return nil, err
			}
			mergedEnvFrom = append(mergedEnvFrom, internalEnvFrom)
		}

	}

	return mergedEnvFrom, nil
}

// mergeVolumeMounts merges given list of VolumeMounts with the volumeMounts
// injected by given podPresets. It returns an error if it detects any conflict during the merge.
func mergeVolumeMounts(volumeMounts []api.VolumeMount, podPresets []*settingsv1alpha1.PodPreset) ([]api.VolumeMount, error) {

	origVolumeMounts := map[string]api.VolumeMount{}
	volumeMountsByPath := map[string]api.VolumeMount{}
	for _, v := range volumeMounts {
		origVolumeMounts[v.Name] = v
		volumeMountsByPath[v.MountPath] = v
	}

	mergedVolumeMounts := make([]api.VolumeMount, len(volumeMounts))
	copy(mergedVolumeMounts, volumeMounts)

	var errs []error

	for _, pp := range podPresets {
		for _, v := range pp.Spec.VolumeMounts {
			internalVolumeMount := api.VolumeMount{}
			if err := apiscorev1.Convert_v1_VolumeMount_To_core_VolumeMount(&v, &internalVolumeMount, nil); err != nil {
				return nil, err
			}
			found, ok := origVolumeMounts[v.Name]
			if !ok {
				// if we don't already have it append it and continue
				origVolumeMounts[v.Name] = internalVolumeMount
				mergedVolumeMounts = append(mergedVolumeMounts, internalVolumeMount)
			} else {
				// make sure they are identical or throw an error
				// shall we throw an error for identical volumeMounts ?
				if !reflect.DeepEqual(found, internalVolumeMount) {
					errs = append(errs, fmt.Errorf("merging volume mounts for %s has a conflict on %s: \n%#v\ndoes not match\n%#v\n in container", pp.GetName(), v.Name, v, found))
				}
			}

			found, ok = volumeMountsByPath[v.MountPath]
			if !ok {
				// if we don't already have it append it and continue
				volumeMountsByPath[v.MountPath] = internalVolumeMount
			} else {
				// make sure they are identical or throw an error
				if !reflect.DeepEqual(found, internalVolumeMount) {
					errs = append(errs, fmt.Errorf("merging volume mounts for %s has a conflict on mount path %s: \n%#v\ndoes not match\n%#v\n in container", pp.GetName(), v.MountPath, v, found))
				}
			}
		}
	}

	err := utilerrors.NewAggregate(errs)
	if err != nil {
		return nil, err
	}

	return mergedVolumeMounts, err
}

// mergeVolumes merges given list of Volumes with the volumes injected by given
// podPresets. It returns an error if it detects any conflict during the merge.
func mergeVolumes(volumes []api.Volume, podPresets []*settingsv1alpha1.PodPreset) ([]api.Volume, error) {
	origVolumes := map[string]api.Volume{}
	for _, v := range volumes {
		origVolumes[v.Name] = v
	}

	mergedVolumes := make([]api.Volume, len(volumes))
	copy(mergedVolumes, volumes)

	var errs []error

	for _, pp := range podPresets {
		for _, v := range pp.Spec.Volumes {
			internalVolume := api.Volume{}
			if err := apiscorev1.Convert_v1_Volume_To_core_Volume(&v, &internalVolume, nil); err != nil {
				return nil, err
			}
			found, ok := origVolumes[v.Name]
			if !ok {
				// if we don't already have it append it and continue
				origVolumes[v.Name] = internalVolume
				mergedVolumes = append(mergedVolumes, internalVolume)
				continue
			}

			// make sure they are identical or throw an error
			if !reflect.DeepEqual(found, internalVolume) {
				errs = append(errs, fmt.Errorf("merging volumes for %s has a conflict on %s: \n%#v\ndoes not match\n%#v\n in container", pp.GetName(), v.Name, v, found))
			}
		}
	}

	err := utilerrors.NewAggregate(errs)
	if err != nil {
		return nil, err
	}

	if len(mergedVolumes) == 0 {
		return nil, nil
	}

	return mergedVolumes, err
}

// applyPodPresetsOnPod updates the PodSpec with merged information from all the
// applicable PodPresets. It ignores the errors of merge functions because merge
// errors have already been checked in safeToApplyPodPresetsOnPod function.
func applyPodPresetsOnPod(pod *api.Pod, podPresets []*settingsv1alpha1.PodPreset) {
	if len(podPresets) == 0 {
		return
	}

	volumes, _ := mergeVolumes(pod.Spec.Volumes, podPresets)
	pod.Spec.Volumes = volumes

	for i, ctr := range pod.Spec.Containers {
		applyPodPresetsOnContainer(&ctr, podPresets)
		pod.Spec.Containers[i] = ctr
	}
	for i, iCtr := range pod.Spec.InitContainers {
		applyPodPresetsOnContainer(&iCtr, podPresets)
		pod.Spec.InitContainers[i] = iCtr
	}

	// add annotation
	if pod.ObjectMeta.Annotations == nil {
		pod.ObjectMeta.Annotations = map[string]string{}
	}

	for _, pp := range podPresets {
		pod.ObjectMeta.Annotations[fmt.Sprintf("%s/podpreset-%s", annotationPrefix, pp.GetName())] = pp.GetResourceVersion()
	}
}

// applyPodPresetsOnContainer injects envVars, VolumeMounts and envFrom from
// given podPresets in to the given container. It ignores conflict errors
// because it assumes those have been checked already by the caller.
func applyPodPresetsOnContainer(ctr *api.Container, podPresets []*settingsv1alpha1.PodPreset) {
	envVars, _ := mergeEnv(ctr.Env, podPresets)
	ctr.Env = envVars

	volumeMounts, _ := mergeVolumeMounts(ctr.VolumeMounts, podPresets)
	ctr.VolumeMounts = volumeMounts

	envFrom, _ := mergeEnvFrom(ctr.EnvFrom, podPresets)
	ctr.EnvFrom = envFrom
}
