/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	batchv1 "k8s.io/api/batch/v1"
	certificatesv1 "k8s.io/api/certificates/v1"
	certificatesv1alpha1 "k8s.io/api/certificates/v1alpha1"
	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	eventsv1beta1 "k8s.io/api/events/v1beta1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/events"
	"k8s.io/kubernetes/pkg/apis/resource"

	// Import install packages to register the types with the legacyscheme
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	_ "k8s.io/kubernetes/pkg/apis/events/install"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
)

// TestSelectableFieldConversion tests field conversion for resources with
// custom selectable fields. All other resources use the default selectable
// fields, so they don't need independent testing.
func TestSelectableFieldConversion(t *testing.T) {
	cases := []struct {
		name            string
		apiVersion      string
		kind            string
		versionedFields fields.Set
		internalFields  fields.Set
		conversions     map[string]string
	}{
		// ----- apps -----
		// No ReplicaSet in apps/v1beta1"
		{
			name:            "replicasets.apps/v1",
			apiVersion:      "apps/v1",
			kind:            "ReplicaSet",
			versionedFields: appsv1.ReplicaSetToSelectableFields(&appsv1.ReplicaSet{}),
			internalFields:  apps.ReplicaSetToSelectableFields(&apps.ReplicaSet{}),
		},
		{
			name:            "replicasets.apps/v1beta2",
			apiVersion:      "apps/v1beta2",
			kind:            "ReplicaSet",
			versionedFields: appsv1beta2.ReplicaSetToSelectableFields(&appsv1beta2.ReplicaSet{}),
			internalFields:  apps.ReplicaSetToSelectableFields(&apps.ReplicaSet{}),
		},
		// ----- batch -----
		{
			name:            "jobs.batch/v1",
			apiVersion:      "batch/v1",
			kind:            "Job",
			versionedFields: batchv1.JobToSelectableFields(&batchv1.Job{}),
			internalFields:  batch.JobToSelectableFields(&batch.Job{}),
		},
		// ----- certificates -----
		// No CertificateSigningRequest in certificates/v1alpha1
		// No ClusterTrustBundle in certificates/v1beta1 or certificates/v1
		{
			name:            "certificatesigningrequests.certificates.k8s.io/v1",
			apiVersion:      "certificates.k8s.io/v1",
			kind:            "CertificateSigningRequest",
			versionedFields: certificatesv1.CertificateSigningRequestToSelectableFields(&certificatesv1.CertificateSigningRequest{}),
			internalFields:  certificates.CertificateSigningRequestToSelectableFields(&certificates.CertificateSigningRequest{}),
		},
		{
			name:            "certificatesigningrequests.certificates.k8s.io/v1beta1",
			apiVersion:      "certificates.k8s.io/v1beta1",
			kind:            "CertificateSigningRequest",
			versionedFields: certificatesv1beta1.CertificateSigningRequestToSelectableFields(&certificatesv1beta1.CertificateSigningRequest{}),
			internalFields:  certificates.CertificateSigningRequestToSelectableFields(&certificates.CertificateSigningRequest{}),
		},
		{
			name:            "clustertrustbundles.certificates.k8s.io/v1alpha1",
			apiVersion:      "certificates.k8s.io/v1alpha1",
			kind:            "ClusterTrustBundle",
			versionedFields: certificatesv1alpha1.ClusterTrustBundleToSelectableFields(&certificatesv1alpha1.ClusterTrustBundle{}),
			internalFields:  certificates.ClusterTrustBundleToSelectableFields(&certificates.ClusterTrustBundle{}),
		},
		// ----- core -----
		{
			name:            "namespaces.core/v1",
			apiVersion:      "v1",
			kind:            "Namespace",
			versionedFields: corev1.NamespaceToSelectableFields(&corev1.Namespace{}),
			internalFields:  core.NamespaceToSelectableFields(&core.Namespace{}),
			conversions:     map[string]string{"name": "metadata.name"},
		},
		{
			name:            "nodes.core/v1",
			apiVersion:      "v1",
			kind:            "Node",
			versionedFields: corev1.NodeToSelectableFields(&corev1.Node{}),
			internalFields:  core.NodeToSelectableFields(&core.Node{}),
		},
		{
			name:            "persistentvolumeclaims.core/v1",
			apiVersion:      "v1",
			kind:            "PersistentVolumeClaim",
			versionedFields: corev1.PersistentVolumeClaimToSelectableFields(&corev1.PersistentVolumeClaim{}),
			internalFields:  core.PersistentVolumeClaimToSelectableFields(&core.PersistentVolumeClaim{}),
			conversions:     map[string]string{"name": "metadata.name"},
		},
		{
			name:            "persistentvolumes.core/v1",
			apiVersion:      "v1",
			kind:            "PersistentVolume",
			versionedFields: corev1.PersistentVolumeToSelectableFields(&corev1.PersistentVolume{}),
			internalFields:  core.PersistentVolumeToSelectableFields(&core.PersistentVolume{}),
			conversions:     map[string]string{"name": "metadata.name"},
		},
		{
			name:            "pods.core/v1",
			apiVersion:      "v1",
			kind:            "Pod",
			versionedFields: corev1.PodToSelectableFields(&corev1.Pod{}),
			internalFields:  core.PodToSelectableFields(&core.Pod{}),
		},
		{
			name:            "replicationcontrollers.core/v1",
			apiVersion:      "v1",
			kind:            "ReplicationController",
			versionedFields: corev1.ReplicationControllerToSelectableFields(&corev1.ReplicationController{}),
			internalFields:  core.ReplicationControllerToSelectableFields(&core.ReplicationController{}),
		},
		{
			name:            "secrets.core/v1",
			apiVersion:      "v1",
			kind:            "Secret",
			versionedFields: corev1.SecretToSelectableFields(&corev1.Secret{}),
			internalFields:  core.SecretToSelectableFields(&core.Secret{}),
		},
		{
			name:            "services.core/v1",
			apiVersion:      "v1",
			kind:            "Service",
			versionedFields: corev1.ServiceToSelectableFields(&corev1.Service{}),
			internalFields:  core.ServiceToSelectableFields(&core.Service{}),
		},
		{
			name:            "events.core/v1",
			apiVersion:      "v1",
			kind:            "Event",
			versionedFields: corev1.EventToSelectableFields(&corev1.Event{}),
			internalFields:  core.EventToSelectableFields(&core.Event{}),
		},
		// ----- events -----
		{
			name:            "events.events.k8s.io/v1",
			apiVersion:      "events.k8s.io/v1",
			kind:            "Event",
			versionedFields: eventsv1.EventToSelectableFields(&eventsv1.Event{}),
			// Migration from core to events is incomplete. Still using core types internally.
			internalFields: events.EventToSelectableFields(&core.Event{}),
			conversions: map[string]string{
				"regarding.kind":            "involvedObject.kind",
				"regarding.namespace":       "involvedObject.namespace",
				"regarding.name":            "involvedObject.name",
				"regarding.uid":             "involvedObject.uid",
				"regarding.apiVersion":      "involvedObject.apiVersion",
				"regarding.resourceVersion": "involvedObject.resourceVersion",
				"regarding.fieldPath":       "involvedObject.fieldPath",
				"reportingController":       "reportingComponent",
			},
		},
		{
			name:            "events.events.k8s.io/v1beta1",
			apiVersion:      "events.k8s.io/v1beta1",
			kind:            "Event",
			versionedFields: eventsv1beta1.EventToSelectableFields(&eventsv1beta1.Event{}),
			// Migration from core to events is incomplete. Still using core types internally.
			internalFields: events.EventToSelectableFields(&core.Event{}),
			conversions: map[string]string{
				"regarding.kind":            "involvedObject.kind",
				"regarding.namespace":       "involvedObject.namespace",
				"regarding.name":            "involvedObject.name",
				"regarding.uid":             "involvedObject.uid",
				"regarding.apiVersion":      "involvedObject.apiVersion",
				"regarding.resourceVersion": "involvedObject.resourceVersion",
				"regarding.fieldPath":       "involvedObject.fieldPath",
				"reportingController":       "reportingComponent",
			},
		},
		// ----- resource -----
		{
			name:            "resourceslices.resource.k8s.io/v1alpha2",
			apiVersion:      "resource.k8s.io/v1alpha2",
			kind:            "ResourceSlice",
			versionedFields: resourcev1alpha2.ResourceSliceToSelectableFields(&resourcev1alpha2.ResourceSlice{}),
			// Migration from core to events is incomplete. Still using core types internally.
			internalFields: resource.ResourceSliceToSelectableFields(&resource.ResourceSlice{}),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			TestSelectableFieldLabelConversionsOfKind(t, legacyscheme.Scheme, tc.apiVersion, tc.kind, tc.versionedFields, tc.internalFields, tc.conversions)
		})
	}
}
