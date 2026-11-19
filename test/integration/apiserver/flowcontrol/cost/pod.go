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

package cost

import (
	"context"
	_ "embed"
	"fmt"
	"runtime"
	"strings"

	"golang.org/x/sync/errgroup"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/ktesting"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	// ephemeralNamespace holds per-scenario pods (committed writes and custom-sized
	// fixtures) that are cleared after each scenario so default stays untouched.
	ephemeralNamespace = "ephemeral"

	acceptJSON     = kruntime.ContentTypeJSON
	acceptProtobuf = kruntime.ContentTypeProtobuf
	acceptCBOR     = kruntime.ContentTypeCBOR
	acceptYAML     = kruntime.ContentTypeYAML
	acceptTable    = "application/json;as=Table;g=meta.k8s.io;v=v1"
)

type contentType struct {
	name   string
	accept string
}

var allReadFormats = []contentType{
	{name: "json", accept: acceptJSON},
	{name: "protobuf", accept: acceptProtobuf},
	{name: "cbor", accept: acceptCBOR},
	{name: "yaml", accept: acceptYAML},
	{name: "table", accept: acceptTable},
}

var watchFormats = []contentType{
	{name: "json", accept: acceptJSON},
	{name: "protobuf", accept: acceptProtobuf},
	{name: "cbor", accept: acceptCBOR},
	{name: "table", accept: acceptTable},
}

var writeFormats = []contentType{
	{name: "json", accept: acceptJSON},
	{name: "protobuf", accept: acceptProtobuf},
	{name: "cbor", accept: acceptCBOR},
}

//go:embed testdata/exemplar_pod.yaml
var exemplarPodYAML []byte

var exemplarPod = mustLoadExemplarPod()

var podCodecs = serializer.NewCodecFactory(
	legacyscheme.Scheme,
	serializer.WithSerializer(cbor.NewSerializerInfo),
)

func mustLoadExemplarPod() *v1.Pod {
	var pod v1.Pod
	if err := yaml.Unmarshal(exemplarPodYAML, &pod); err != nil {
		panic(fmt.Sprintf("failed to unmarshal exemplar_pod.yaml: %v", err))
	}
	pod.Namespace = ""
	pod.Name = ""
	pod.UID = ""
	pod.ResourceVersion = ""
	pod.CreationTimestamp = metav1.Time{}
	pod.ManagedFields = nil
	pod.Spec.NodeName = ""
	return &pod
}

func encodeObject(tb testutils.TB, mediaType string, obj kruntime.Object) []byte {
	tb.Helper()
	if mediaType == "" || mediaType == acceptTable {
		mediaType = acceptJSON
	}
	info, ok := kruntime.SerializerInfoForMediaType(podCodecs.SupportedMediaTypes(), mediaType)
	if !ok {
		tb.Fatalf("serializer not found for %s", mediaType)
	}
	gvks, _, err := legacyscheme.Scheme.ObjectKinds(obj)
	if err != nil || len(gvks) == 0 {
		tb.Fatalf("failed to determine GVK for %T: %v", obj, err)
	}
	gv := gvks[0].GroupVersion()
	codec := podCodecs.CodecForVersions(info.Serializer, podCodecs.UniversalDeserializer(), gv, gv)
	data, err := kruntime.Encode(codec, obj)
	if err != nil {
		tb.Fatalf("failed to encode %T as %s: %v", obj, mediaType, err)
	}
	return data
}

func ensureNamespace(tCtx ktesting.TContext, ns string) {
	tCtx.Helper()
	if ns != metav1.NamespaceDefault {
		if _, err := tCtx.Client().CoreV1().Namespaces().Create(tCtx, &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{Name: ns},
		}, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
			tCtx.Fatalf("failed to create namespace %s: %v", ns, err)
		}
	}
	if _, err := tCtx.Client().CoreV1().ServiceAccounts(ns).Create(tCtx, &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: ns},
	}, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		tCtx.Fatalf("failed to create default ServiceAccount in %s: %v", ns, err)
	}
}

// seedExemplarPod creates a pod and then updates its status subresource.
// kube-apiserver's podStrategy.PrepareForCreate resets pod.Status on creation,
// so a second write to /status is required to populate the exemplar's status
// payload and kubelet managedFields entry (~10KB total stored size).
func seedExemplarPod(ctx context.Context, client clientset.Interface, ns, name, nodeName, deploymentLabel string, extraBytes int) (*v1.Pod, error) {
	pod := exemplarPod.DeepCopy()
	pod.Namespace = ns
	pod.Name = name
	pod.Spec.NodeName = nodeName
	if deploymentLabel != "" {
		pod.Labels["name"] = deploymentLabel
	}
	if extraBytes > 0 {
		pod.Annotations["bench.k8s.io/payload"] = strings.Repeat("x", extraBytes)
	}

	created, err := client.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{FieldManager: "seed-client"})
	if err != nil {
		return nil, fmt.Errorf("seed create %s/%s failed: %w", ns, name, err)
	}
	created.Status = exemplarPod.Status
	updated, err := client.CoreV1().Pods(ns).UpdateStatus(ctx, created, metav1.UpdateOptions{FieldManager: "kubelet"})
	if err != nil {
		return nil, fmt.Errorf("seed status update %s/%s failed: %w", ns, name, err)
	}
	return updated, nil
}

func seedPods(tCtx ktesting.TContext, informer *informerWaiter, ns string, count, extraBytes int) []*v1.Pod {
	tCtx.Helper()
	informer.resetEvents()
	client := tCtx.Client()
	pods := make([]*v1.Pod, count)
	seedAt := func(ctx context.Context, i int) error {
		var nodeName string
		// Leave the final pod unbound when seeding multiple pods so dry-run
		// POST /binding can target it.
		if count == 1 || i < count-1 {
			nodeName = fmt.Sprintf("kind-worker%d", i%10)
		}
		deploymentLabel := fmt.Sprintf("small-deployment-%02d", i%10)
		extra := extraBytes
		if extra > 0 {
			extra += i
		}
		p, err := seedExemplarPod(ctx, client, ns, fmt.Sprintf("pod-%03d", i), nodeName, deploymentLabel, extra)
		pods[i] = p
		return err
	}

	// Seed pod-000 first and the final pod last so pods[0] and pods[count-1]
	// bracket the ResourceVersion range for FIRST_RV and LATEST_RV without sorting.
	if count > 0 {
		if err := seedAt(tCtx, 0); err != nil {
			tCtx.Fatalf("%v", err)
		}
		g, ctx := errgroup.WithContext(tCtx)
		g.SetLimit(runtime.GOMAXPROCS(0))
		for i := 1; i < count-1; i++ {
			g.Go(func() error { return seedAt(ctx, i) })
		}
		if err := g.Wait(); err != nil {
			tCtx.Fatalf("%v", err)
		}
		if count > 1 {
			if err := seedAt(tCtx, count-1); err != nil {
				tCtx.Fatalf("%v", err)
			}
		}
	}
	// Each seedExemplarPod performs Create + UpdateStatus (2 informer events).
	informer.wait(tCtx, tCtx, 2*int64(count))
	informer.resetEvents()
	return pods
}

func clearNamespace(tCtx ktesting.TContext, informer *informerWaiter, ns string) {
	tCtx.Helper()
	items, err := informer.informer.GetIndexer().ByIndex(cache.NamespaceIndex, ns)
	if err != nil {
		tCtx.Fatalf("failed to query pod informer namespace index for %s: %v", ns, err)
	}
	if len(items) == 0 {
		informer.resetEvents()
		return
	}
	informer.resetEvents()
	if err := tCtx.Client().CoreV1().Pods(ns).DeleteCollection(
		tCtx,
		*metav1.NewDeleteOptions(0),
		metav1.ListOptions{},
	); err != nil {
		tCtx.Fatalf("failed to clear pods in %s: %v", ns, err)
	}
	informer.wait(tCtx, tCtx, int64(len(items)))
	informer.resetEvents()
}

type bodyBuilder func(mediaType string, pod *v1.Pod, alt int) []byte

func staticPodBody(tb testutils.TB) bodyBuilder {
	tb.Helper()
	return staticObjectBody(tb, exemplarPod.DeepCopy())
}

func staticObjectBody(tb testutils.TB, obj kruntime.Object) bodyBuilder {
	tb.Helper()
	encodedByType := make(map[string][]byte, len(writeFormats))
	for _, wf := range writeFormats {
		encodedByType[wf.accept] = encodeObject(tb, wf.accept, obj)
	}
	return func(mediaType string, _ *v1.Pod, _ int) []byte {
		return encodedByType[mediaType]
	}
}

func staticBytesBody(payloads ...string) bodyBuilder {
	raw := make([][]byte, len(payloads))
	for i, p := range payloads {
		raw[i] = []byte(p)
	}
	return func(_ string, _ *v1.Pod, alt int) []byte {
		return raw[alt%len(raw)]
	}
}
