/*
Copyright 2021 The Kubernetes Authors.

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

package podsecurity

import (
	"context"
	"io/ioutil"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/core"
	v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/features"
	podsecurityadmission "k8s.io/pod-security-admission/admission"
	"sigs.k8s.io/yaml"
)

func TestConvert(t *testing.T) {
	extractor := podsecurityadmission.DefaultPodSpecExtractor{}
	internalTypes := map[schema.GroupResource]runtime.Object{
		core.Resource("pods"):                   &core.Pod{},
		core.Resource("replicationcontrollers"): &core.ReplicationController{},
		core.Resource("podtemplates"):           &core.PodTemplate{},
		apps.Resource("replicasets"):            &apps.ReplicaSet{},
		apps.Resource("deployments"):            &apps.Deployment{},
		apps.Resource("statefulsets"):           &apps.StatefulSet{},
		apps.Resource("daemonsets"):             &apps.DaemonSet{},
		batch.Resource("jobs"):                  &batch.Job{},
		batch.Resource("cronjobs"):              &batch.CronJob{},
	}
	for _, r := range extractor.PodSpecResources() {
		internalType, ok := internalTypes[r]
		if !ok {
			t.Errorf("no internal type registered for %s", r.String())
			continue
		}
		externalType, err := convert(internalType)
		if err != nil {
			t.Errorf("error converting %T: %v", internalType, err)
			continue
		}
		_, _, err = extractor.ExtractPodSpec(externalType)
		if err != nil {
			t.Errorf("error extracting from %T: %v", externalType, err)
			continue
		}
	}
}

func BenchmarkVerifyPod(b *testing.B) {
	defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.PodSecurity, true)()

	p, err := newPlugin(nil)
	if err != nil {
		b.Fatal(err)
	}

	p.InspectFeatureGates(utilfeature.DefaultFeatureGate)

	enforceImplicitPrivilegedNamespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "enforce-implicit", Labels: map[string]string{}}}
	enforcePrivilegedNamespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "enforce-privileged", Labels: map[string]string{"pod-security.kubernetes.io/enforce": "privileged"}}}
	enforceBaselineNamespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "enforce-baseline", Labels: map[string]string{"pod-security.kubernetes.io/enforce": "baseline"}}}
	enforceRestrictedNamespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "enforce-restricted", Labels: map[string]string{"pod-security.kubernetes.io/enforce": "restricted"}}}
	warnBaselineNamespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "warn-baseline", Labels: map[string]string{"pod-security.kubernetes.io/warn": "baseline"}}}
	warnRestrictedNamespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "warn-restricted", Labels: map[string]string{"pod-security.kubernetes.io/warn": "restricted"}}}
	enforceWarnAuditBaseline := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "enforce-warn-audit-baseline", Labels: map[string]string{"pod-security.kubernetes.io/enforce": "baseline", "pod-security.kubernetes.io/warn": "baseline", "pod-security.kubernetes.io/audit": "baseline"}}}
	warnBaselineAuditRestrictedNamespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "warn-baseline-audit-restricted", Labels: map[string]string{"pod-security.kubernetes.io/warn": "baseline", "pod-security.kubernetes.io/audit": "restricted"}}}
	c := fake.NewSimpleClientset(
		enforceImplicitPrivilegedNamespace,
		enforcePrivilegedNamespace,
		enforceBaselineNamespace,
		enforceRestrictedNamespace,
		warnBaselineNamespace,
		warnRestrictedNamespace,
		enforceWarnAuditBaseline,
		warnBaselineAuditRestrictedNamespace,
	)
	p.SetExternalKubeClientSet(c)

	informerFactory := informers.NewSharedInformerFactory(c, 0)
	p.SetExternalKubeInformerFactory(informerFactory)
	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)
	informerFactory.WaitForCacheSync(stopCh)

	if err := p.ValidateInitialization(); err != nil {
		b.Fatal(err)
	}

	corePod := &core.Pod{}
	v1Pod := &corev1.Pod{}
	data, err := ioutil.ReadFile("testdata/pod.yaml")
	if err != nil {
		b.Fatal(err)
	}
	if err := yaml.Unmarshal(data, v1Pod); err != nil {
		b.Fatal(err)
	}
	if err := v1.Convert_v1_Pod_To_core_Pod(v1Pod, corePod, nil); err != nil {
		b.Fatal(err)
	}

	appsDeployment := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "mydeployment"},
		Spec: apps.DeploymentSpec{
			Template: core.PodTemplateSpec{
				ObjectMeta: corePod.ObjectMeta,
				Spec:       corePod.Spec,
			},
		},
	}

	namespaces := []string{
		"enforce-implicit", "enforce-privileged", "enforce-baseline", "enforce-restricted",
		"warn-baseline", "warn-restricted",
		"enforce-warn-audit-baseline", "warn-baseline-audit-restricted",
	}
	for _, namespace := range namespaces {
		b.Run(namespace+"_pod", func(b *testing.B) {
			ctx := context.Background()
			attrs := admission.NewAttributesRecord(
				corePod.DeepCopy(), nil,
				schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
				namespace, "mypod",
				schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
				"",
				admission.Create, &metav1.CreateOptions{}, false,
				&user.DefaultInfo{Name: "myuser"},
			)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := p.Validate(ctx, attrs, nil); err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run(namespace+"_deployment", func(b *testing.B) {
			ctx := context.Background()
			attrs := admission.NewAttributesRecord(
				appsDeployment.DeepCopy(), nil,
				schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
				namespace, "mydeployment",
				schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
				"",
				admission.Create, &metav1.CreateOptions{}, false,
				&user.DefaultInfo{Name: "myuser"},
			)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := p.Validate(ctx, attrs, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
