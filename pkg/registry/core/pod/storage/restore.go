/*
Copyright The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"
	"slices"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	nodeclient "k8s.io/client-go/kubernetes/typed/node/v1alpha1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
)

type restorePolicy struct {
	authorizer  authorizer.UnconditionalAuthorizer
	checkpoints nodeclient.PodCheckpointsGetter
}

func (r *REST) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	return r.Store.Create(ctx, obj, r.restorePolicy.createValidation(createValidation), options)
}

func (r *REST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// Update also handles apply-on-create, which must enforce the same restore checks as POST.
	return r.Store.Update(ctx, name, objInfo, r.restorePolicy.createValidation(createValidation), r.restorePolicy.updateValidation(updateValidation), forceAllowCreate, options)
}

func (p *restorePolicy) createValidation(validate rest.ValidateObjectFunc) rest.ValidateObjectFunc {
	return func(ctx context.Context, obj runtime.Object) error {
		if err := p.validateCreate(ctx, obj.(*api.Pod)); err != nil {
			return err
		}
		if validate != nil {
			return validate(ctx, obj)
		}
		return nil
	}
}

func (p *restorePolicy) updateValidation(validate rest.ValidateObjectUpdateFunc) rest.ValidateObjectUpdateFunc {
	return func(ctx context.Context, obj, old runtime.Object) error {
		if err := p.validateUpdate(ctx, obj.(*api.Pod), old.(*api.Pod)); err != nil {
			return err
		}
		if validate != nil {
			return validate(ctx, obj, old)
		}
		return nil
	}
}

func (p *restorePolicy) validateCreate(ctx context.Context, pod *api.Pod) error {
	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelCheckpointRestore) || pod.Spec.RestoreFrom == nil {
		return nil
	}
	checkpointName := pod.Spec.RestoreFrom.Name
	attrs := authorizer.AttributesRecord{
		Verb: "restore", APIGroup: "node.k8s.io", APIVersion: "v1alpha1",
		Resource: "podcheckpoints", Name: checkpointName, ResourceRequest: true,
	}
	// Authorize the final reference before reading it, including when mutation admission is disabled.
	if err := p.authorize(ctx, pod, attrs); err != nil {
		return err
	}
	if p.checkpoints == nil {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("restore requires a PodCheckpoint client"))
	}
	namespace, _ := genericapirequest.NamespaceFrom(ctx)
	checkpoint, err := p.checkpoints.PodCheckpoints(namespace).Get(ctx, checkpointName, metav1.GetOptions{})
	if err != nil {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("failed to read PodCheckpoint %q referenced by spec.restoreFrom: %w", checkpointName, err))
	}
	if err := ctx.Err(); err != nil {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, err)
	}
	if pod.Spec.RestoreFrom.UID == "" || pod.Spec.RestoreFrom.UID != checkpoint.UID {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("spec.restoreFrom.uid must match PodCheckpoint %q UID %q", checkpointName, checkpoint.UID))
	}
	if checkpoint.DeletionTimestamp != nil {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("PodCheckpoint %q is being deleted", checkpointName))
	}
	if !apimeta.IsStatusConditionTrue(checkpoint.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady) {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("PodCheckpoint %q is not ready; restore requires status.conditions Ready=True", checkpointName))
	}
	if pod.Spec.NodeName != "" {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("pod restoring from PodCheckpoint %q must not set spec.nodeName; restore placement is controlled via node affinity", checkpointName))
	}
	if !slices.Contains(pod.Finalizers, nodeapi.PodRestoreProtectionFinalizer) {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("pod restoring from PodCheckpoint %q must retain the %q finalizer until runtime cleanup is confirmed", checkpointName, nodeapi.PodRestoreProtectionFinalizer))
	}
	return validatePodSpecMatchesCheckpoint(pod, checkpointName, checkpoint)
}

func (p *restorePolicy) validateUpdate(ctx context.Context, pod, oldPod *api.Pod) error {
	// Stored protection outlives both the feature gate and the restore reference.
	if !slices.Contains(oldPod.Finalizers, nodeapi.PodRestoreProtectionFinalizer) || slices.Contains(pod.Finalizers, nodeapi.PodRestoreProtectionFinalizer) {
		return nil
	}
	return p.authorize(ctx, pod, authorizer.AttributesRecord{
		Verb: "update", APIGroup: "", APIVersion: "v1", Resource: "pods",
		Subresource: "finalizers", Name: oldPod.Name, ResourceRequest: true,
	})
}

func (p *restorePolicy) authorize(ctx context.Context, pod *api.Pod, attrs authorizer.AttributesRecord) error {
	forbidden := func(err error) error { return apierrors.NewForbidden(api.Resource("pods"), pod.Name, err) }
	resource := attrs.Resource
	if attrs.Subresource != "" {
		resource += "/" + attrs.Subresource
	}
	if err := ctx.Err(); err != nil {
		return forbidden(err)
	}
	if p == nil || p.authorizer == nil {
		return forbidden(fmt.Errorf("%s on %s requires an authorizer", attrs.Verb, resource))
	}
	user, ok := genericapirequest.UserFrom(ctx)
	if !ok || user == nil {
		return forbidden(fmt.Errorf("%s on %s requires a user in the request context", attrs.Verb, resource))
	}
	namespace, ok := genericapirequest.NamespaceFrom(ctx)
	if !ok || namespace == "" {
		return forbidden(fmt.Errorf("%s on %s requires a namespace in the request context", attrs.Verb, resource))
	}
	attrs.User, attrs.Namespace = user, namespace
	decision, reason, err := p.authorizer.Authorize(ctx, attrs)
	if err != nil {
		return forbidden(fmt.Errorf("error authorizing %s on %s: %w", attrs.Verb, resource, err))
	}
	if decision != authorizer.DecisionAllow {
		return forbidden(fmt.Errorf("%s on %s %q in namespace %q is not authorized: %s", attrs.Verb, resource, attrs.Name, namespace, reason))
	}
	if err := ctx.Err(); err != nil {
		return forbidden(err)
	}
	return nil
}

// Compare the captured scheduling constraints with the same node pin added by admission.
// Sanitization excludes nodeName, restoreFrom, ephemeral containers and scheduling gates.
func validatePodSpecMatchesCheckpoint(pod *api.Pod, checkpointName string, checkpoint *nodev1alpha1.PodCheckpoint) error {
	tmpl := checkpoint.Status.CheckpointedPodTemplate
	if tmpl == nil {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("PodCheckpoint %q has no status.checkpointedPodTemplate; restore requires a captured pod template", checkpointName))
	}
	var live v1.Pod
	if err := legacyscheme.Scheme.Convert(pod, &live, nil); err != nil {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("failed to convert pod for the PodCheckpoint %q equality check: %w", checkpointName, err))
	}
	if checkpoint.Status.NodeName == nil || *checkpoint.Status.NodeName == "" {
		return apierrors.NewForbidden(api.Resource("pods"), pod.Name, fmt.Errorf("PodCheckpoint %q has no status.nodeName", checkpointName))
	}
	want := tmpl.Spec.DeepCopy()
	checkpointutil.AddRestoreNodeAffinity(want, *checkpoint.Status.NodeName)
	got := checkpointutil.SanitizePodTemplate(&live).Spec
	if changedFields := checkpointutil.PodSpecChangedFields(want, &got); len(changedFields) != 0 {
		errs := make(field.ErrorList, 0, len(changedFields))
		for _, path := range changedFields {
			errs = append(errs, field.Forbidden(field.NewPath(path), fmt.Sprintf("must match the pod spec captured in PodCheckpoint %q", checkpointName)))
		}
		return apierrors.NewInvalid(schema.GroupKind{Kind: "Pod"}, pod.Name, errs)
	}
	return nil
}
