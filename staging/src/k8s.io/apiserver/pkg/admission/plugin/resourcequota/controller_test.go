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

package resourcequota

import (
	"errors"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	resourcequotaapi "k8s.io/apiserver/pkg/admission/plugin/resourcequota/apis/resourcequota"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
)

// conflictOnceQuotaAccessor fails the first status update of conflictQuota with a conflict error.
type conflictOnceQuotaAccessor struct {
	quotas        []corev1.ResourceQuota
	conflictQuota string
	conflicted    bool
}

func (a *conflictOnceQuotaAccessor) UpdateQuotaStatus(newQuota *corev1.ResourceQuota) error {
	if newQuota.Name == a.conflictQuota && !a.conflicted {
		a.conflicted = true
		return apierrors.NewConflict(corev1.Resource("resourcequotas"), newQuota.Name, errors.New("stale resource version"))
	}
	return nil
}

func (a *conflictOnceQuotaAccessor) GetQuotas(namespace string) ([]corev1.ResourceQuota, error) {
	return a.quotas, nil
}

// TestCheckQuotasPreservesDenialAcrossConflictRetry covers https://github.com/kubernetes/kubernetes/issues/141904:
// a request denied in the first pass must stay denied when a conflict on an unrelated quota
// in the same batch triggers a retry that re-evaluates only the conflicted quota.
func TestCheckQuotasPreservesDenialAcrossConflictRetry(t *testing.T) {
	const namespace = "test"
	quotas := []corev1.ResourceQuota{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "secret-quota", Namespace: namespace},
			Status: corev1.ResourceQuotaStatus{
				Hard: corev1.ResourceList{corev1.ResourceSecrets: resource.MustParse("1")},
				Used: corev1.ResourceList{corev1.ResourceSecrets: resource.MustParse("1")},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "configmap-quota", Namespace: namespace},
			Status: corev1.ResourceQuotaStatus{
				Hard: corev1.ResourceList{corev1.ResourceConfigMaps: resource.MustParse("10")},
				Used: corev1.ResourceList{corev1.ResourceConfigMaps: resource.MustParse("0")},
			},
		},
	}

	evaluator := &quotaEvaluator{
		quotaAccessor: &conflictOnceQuotaAccessor{quotas: quotas, conflictQuota: "configmap-quota"},
		registry: generic.NewRegistry([]quota.Evaluator{
			generic.NewObjectCountEvaluator(corev1.Resource("secrets"), nil, corev1.ResourceSecrets),
			generic.NewObjectCountEvaluator(corev1.Resource("configmaps"), nil, corev1.ResourceConfigMaps),
		}),
		config: &resourcequotaapi.Configuration{},
	}

	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "over-quota", Namespace: namespace}}
	configMap := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "within-quota", Namespace: namespace}}
	deniedWaiter := newAdmissionWaiter(admission.NewAttributesRecord(secret, nil, corev1.SchemeGroupVersion.WithKind("Secret"), namespace, secret.Name, corev1.SchemeGroupVersion.WithResource("secrets"), "", admission.Create, &metav1.CreateOptions{}, false, nil))
	admittedWaiter := newAdmissionWaiter(admission.NewAttributesRecord(configMap, nil, corev1.SchemeGroupVersion.WithKind("ConfigMap"), namespace, configMap.Name, corev1.SchemeGroupVersion.WithResource("configmaps"), "", admission.Create, &metav1.CreateOptions{}, false, nil))

	evaluator.checkQuotas(quotas, []*admissionWaiter{deniedWaiter, admittedWaiter}, 3)

	if !apierrors.IsForbidden(deniedWaiter.result) {
		t.Errorf("expected over-quota request to stay forbidden after conflict retry, got: %v", deniedWaiter.result)
	}
	if admittedWaiter.result != nil {
		t.Errorf("expected within-quota request to be admitted, got: %v", admittedWaiter.result)
	}
}
