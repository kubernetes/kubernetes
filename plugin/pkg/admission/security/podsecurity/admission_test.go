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
	"fmt"
	"io/ioutil"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/warning"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/core"
	v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	podsecurityadmission "k8s.io/pod-security-admission/admission"
	"k8s.io/utils/pointer"
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
	data, err := ioutil.ReadFile("testdata/pod_restricted.yaml")
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

func BenchmarkVerifyNamespace(b *testing.B) {
	p, err := newPlugin(nil)
	if err != nil {
		b.Fatal(err)
	}

	p.InspectFeatureGates(utilfeature.DefaultFeatureGate)

	namespace := "enforce"
	enforceNamespaceBaselineV1 := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace, Labels: map[string]string{"pod-security.kubernetes.io/enforce": "baseline"}}}
	enforceNamespaceRestrictedV1 := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace, Labels: map[string]string{"pod-security.kubernetes.io/enforce": "restricted"}}}

	enforceNamespaceBaselineCore := &core.Namespace{}
	if err := v1.Convert_v1_Namespace_To_core_Namespace(enforceNamespaceBaselineV1, enforceNamespaceBaselineCore, nil); err != nil {
		b.Fatal(err)
	}
	enforceNamespaceRestrictedCore := &core.Namespace{}
	if err := v1.Convert_v1_Namespace_To_core_Namespace(enforceNamespaceRestrictedV1, enforceNamespaceRestrictedCore, nil); err != nil {
		b.Fatal(err)
	}

	v1Pod := &corev1.Pod{}
	data, err := ioutil.ReadFile("testdata/pod_baseline.yaml")
	if err != nil {
		b.Fatal(err)
	}
	if err := yaml.Unmarshal(data, v1Pod); err != nil {
		b.Fatal(err)
	}

	// https://github.com/kubernetes/community/blob/master/sig-scalability/configs-and-limits/thresholds.md#kubernetes-thresholds
	ownerA := metav1.OwnerReference{
		APIVersion: "apps/v1",
		Kind:       "ReplicaSet",
		Name:       "myapp-123123",
		UID:        types.UID("7610a7f4-8f80-4f88-95b5-6cefdd8e9dbd"),
		Controller: pointer.Bool(true),
	}
	ownerB := metav1.OwnerReference{
		APIVersion: "apps/v1",
		Kind:       "ReplicaSet",
		Name:       "myapp-234234",
		UID:        types.UID("7610a7f4-8f80-4f88-95b5-as765as76f55"),
		Controller: pointer.Bool(true),
	}

	// number of warnings printed for the entire namespace
	namespaceWarningCount := 1

	podCount := 3000
	objects := make([]runtime.Object, 0, podCount+1)
	objects = append(objects, enforceNamespaceBaselineV1)
	for i := 0; i < podCount; i++ {
		v1PodCopy := v1Pod.DeepCopy()
		v1PodCopy.Name = fmt.Sprintf("pod%d", i)
		v1PodCopy.UID = types.UID(fmt.Sprintf("pod%d", i))
		v1PodCopy.Namespace = namespace
		switch i % 3 {
		case 0:
			v1PodCopy.OwnerReferences = []metav1.OwnerReference{ownerA}
		case 1:
			v1PodCopy.OwnerReferences = []metav1.OwnerReference{ownerB}
		default:
			// no owner references
		}
		objects = append(objects, v1PodCopy)
	}

	c := fake.NewSimpleClientset(
		objects...,
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

	ctx := context.Background()
	attrs := admission.NewAttributesRecord(
		enforceNamespaceRestrictedCore.DeepCopy(), enforceNamespaceBaselineCore.DeepCopy(),
		schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Namespace"},
		namespace, namespace,
		schema.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"},
		"",
		admission.Update, &metav1.UpdateOptions{}, false,
		&user.DefaultInfo{Name: "myuser"},
	)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dc := dummyRecorder{agent: "", text: ""}
		ctxWithRecorder := warning.WithWarningRecorder(ctx, &dc)
		if err := p.Validate(ctxWithRecorder, attrs, nil); err != nil {
			b.Fatal(err)
		}
		// should either be a single aggregated warning, or a unique warning per pod
		if dc.count != (1+namespaceWarningCount) && dc.count != (podCount+namespaceWarningCount) {
			b.Fatalf("expected either %d or %d warnings, got %d", 1+namespaceWarningCount, podCount+namespaceWarningCount, dc.count)
		}
		// warning should contain the runAsNonRoot issue
		if e, a := "runAsNonRoot", dc.text; !strings.Contains(a, e) {
			b.Fatalf("expected warning containing %q, got %q", e, a)
		}
	}
}

type dummyRecorder struct {
	count int
	agent string
	text  string
}

func (r *dummyRecorder) AddWarning(agent, text string) {
	r.count++
	r.agent = agent
	r.text = text
	return
}

var _ warning.Recorder = &dummyRecorder{}
