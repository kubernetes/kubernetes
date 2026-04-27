/*
Copyright 2019 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"fmt"

	g "github.com/onsi/ginkgo/v2"
	o "github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/kubernetes"
	aggregatorclientset "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	samplev1alpha1 "k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"
	samplev1alpha1client "k8s.io/sample-apiserver/pkg/generated/clientset/versioned/typed/wardle/v1alpha1"
)

var _ = SIGDescribe("client-go should negotiate", func() {
	f := framework.NewDefaultFramework("protocol")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	for _, s := range []string{
		"application/json",
		"application/vnd.kubernetes.protobuf",
		"application/vnd.kubernetes.protobuf,application/json",
		"application/json,application/vnd.kubernetes.protobuf",
	} {
		accept := s
		g.It(fmt.Sprintf("watch and report errors with accept %q", accept), func() {
			g.By("creating an object for which we will watch")
			ns := f.Namespace.Name
			client := f.ClientSet.CoreV1().ConfigMaps(ns)
			configMapName := "e2e-client-go-test-negotiation"
			testConfigMap := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: configMapName}}
			before, err := client.List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			_, err = client.Create(context.TODO(), testConfigMap, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			opts := metav1.ListOptions{
				ResourceVersion: before.ResourceVersion,
				FieldSelector:   fields.SelectorFromSet(fields.Set{"metadata.name": configMapName}).String(),
			}

			g.By("watching for changes on the object")
			cfg, err := framework.LoadConfig()
			framework.ExpectNoError(err)

			cfg.AcceptContentTypes = accept

			c := kubernetes.NewForConfigOrDie(cfg)
			w, err := c.CoreV1().ConfigMaps(ns).Watch(context.TODO(), opts)
			framework.ExpectNoError(err)
			defer w.Stop()

			evt, ok := <-w.ResultChan()
			o.Expect(ok).To(o.BeTrueBecause("unexpected watch event: %v, %#v", evt.Type, evt.Object))
			switch evt.Type {
			case watch.Added, watch.Modified:
				// this is allowed
			case watch.Error:
				err := apierrors.FromObject(evt.Object)
				// In Kubernetes 1.17 and earlier, the api server returns both apierrors.StatusReasonExpired and
				// apierrors.StatusReasonGone for HTTP 410 (Gone) status code responses. In 1.18 the kube server is more consistent
				// and always returns apierrors.StatusReasonExpired. For backward compatibility we can only remove the apierrs.IsGone
				// check when we fully drop support for Kubernetes 1.17 servers from reflectors.
				if apierrors.IsGone(err) || apierrors.IsResourceExpired(err) {
					// this is allowed, since the kubernetes object could be very old
					break
				}
				if apierrors.IsUnexpectedObjectError(err) {
					g.Fail(fmt.Sprintf("unexpected object, wanted v1.Status: %#v", evt.Object))
				}
				g.Fail(fmt.Sprintf("unexpected error: %#v", evt.Object))
			default:
				g.Fail(fmt.Sprintf("unexpected type %s: %#v", evt.Type, evt.Object))
			}
		})
	}
})

var _ = SIGDescribe("CBOR", feature.CBOR, func() {
	f := framework.NewDefaultFramework("cbor")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	// Must be serial to avoid conflict with other tests that set up a sample apiserver.
	f.It("clients remain compatible with the 1.17 sample-apiserver", f.WithSerial(), func(ctx context.Context) {
		clientfeaturestesting.SetFeatureDuringTest(g.GinkgoTB(), clientfeatures.ClientsAllowCBOR, true)
		clientfeaturestesting.SetFeatureDuringTest(g.GinkgoTB(), clientfeatures.ClientsPreferCBOR, true)

		clientConfig, err := framework.LoadConfig()
		framework.ExpectNoError(err)

		aggregatorClient, err := aggregatorclientset.NewForConfig(clientConfig)
		framework.ExpectNoError(err)

		dynamicClient, err := dynamic.NewForConfig(clientConfig)
		framework.ExpectNoError(err)

		objectNames := generateSampleAPIServerObjectNames(f.Namespace.Name)
		g.DeferCleanup(func(ctx context.Context) {
			cleanupSampleAPIServer(ctx, f.ClientSet, aggregatorClient, objectNames, samplev1alpha1.SchemeGroupVersion.Version+"."+samplev1alpha1.SchemeGroupVersion.Group)
		})
		SetUpSampleAPIServer(ctx, f, aggregatorClient, imageutils.GetE2EImage(imageutils.APIServer), objectNames, samplev1alpha1.SchemeGroupVersion.Group, samplev1alpha1.SchemeGroupVersion.Version)

		flunder := samplev1alpha1.Flunder{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-flunder",
			},
		}

		g.By("making requests with a generated client", func() {
			sampleClient, err := samplev1alpha1client.NewForConfig(clientConfig)
			framework.ExpectNoError(err)

			_, err = sampleClient.Flunders(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: "a,!a"})
			framework.ExpectNoError(err, "Failed to list with generated client")

			_, err = sampleClient.Flunders(f.Namespace.Name).Create(ctx, &flunder, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			o.Expect(err).To(o.MatchError(apierrors.IsUnsupportedMediaType, "Expected 415 (Unsupported Media Type) response on first write with generated client"))

			_, err = sampleClient.Flunders(f.Namespace.Name).Create(ctx, &flunder, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			framework.ExpectNoError(err, "Expected subsequent writes to succeed with generated client")
		})

		g.By("making requests with a dynamic client", func() {
			unstructuredFlunderContent, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&flunder)
			framework.ExpectNoError(err)
			unstructuredFlunder := &unstructured.Unstructured{Object: unstructuredFlunderContent}

			flunderDynamicClient := dynamicClient.Resource(samplev1alpha1.SchemeGroupVersion.WithResource("flunders")).Namespace(f.Namespace.Name)

			list, err := flunderDynamicClient.List(ctx, metav1.ListOptions{LabelSelector: "a,!a"})
			framework.ExpectNoError(err, "Failed to list with dynamic client")
			o.Expect(list.GetObjectKind().GroupVersionKind()).To(o.Equal(samplev1alpha1.SchemeGroupVersion.WithKind("FlunderList")))

			_, err = flunderDynamicClient.Create(ctx, unstructuredFlunder, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			o.Expect(err).To(o.MatchError(apierrors.IsUnsupportedMediaType, "Expected 415 (Unsupported Media Type) response on first write with dynamic client"))

			_, err = flunderDynamicClient.Create(ctx, unstructuredFlunder, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			framework.ExpectNoError(err, "Expected subsequent writes to succeed with dynamic client")
		})
	})
})
