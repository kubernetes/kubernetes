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

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInformerTrimCodegen(t *testing.T) {
	update := os.Getenv("UPDATE_INFORMER_TRIM") == "true"
	for _, cfg := range DefaultConfigs(filepath.Join("..", "..", "..")) {
		t.Run(filepath.Base(cfg.BinaryPkg), func(t *testing.T) {
			Verify(t, cfg, update)
		})
	}
}

type captureReporter struct {
	t      testing.TB
	errors []string
}

func (r *captureReporter) Errorf(format string, args ...any) {
	r.errors = append(r.errors, fmt.Sprintf(format, args...))
}

func (r *captureReporter) Fatalf(format string, args ...any) {
	r.t.Fatalf(format, args...)
}

func writeSyntheticModule(t *testing.T, extraAppBody string) Config {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"go.mod": strings.Join([]string{
			"module k8s.io/synthetic",
			"",
			"go 1.24",
			"",
			"require (",
			"\tk8s.io/api v0.0.0",
			"\tk8s.io/apimachinery v0.0.0",
			"\tk8s.io/apiserver v0.0.0",
			"\tk8s.io/client-go v0.0.0",
			")",
			"",
		}, "\n"),
		"vendor/modules.txt": strings.Join([]string{
			"# k8s.io/apimachinery v0.0.0",
			"## explicit",
			"k8s.io/apimachinery/pkg/apis/meta/v1",
			"k8s.io/apimachinery/pkg/runtime/schema",
			"# k8s.io/api v0.0.0",
			"## explicit",
			"k8s.io/api/core/v1",
			"# k8s.io/client-go v0.0.0",
			"## explicit",
			"k8s.io/client-go/informers",
			"k8s.io/client-go/informers/core/v1",
			"k8s.io/client-go/listers/core/v1",
			"k8s.io/client-go/kubernetes/typed/core/v1",
			"# k8s.io/apiserver v0.0.0",
			"## explicit",
			"k8s.io/apiserver/pkg/quota/v1",
			"k8s.io/apiserver/pkg/quota/v1/generic",
			"",
		}, "\n"),
		"vendor/k8s.io/apimachinery/pkg/apis/meta/v1/types.go": `package v1
type ObjectMeta struct {
	Name            string
	GenerateName    string
	Namespace       string
	SelfLink        string
	UID             string
	ResourceVersion string
}
`,
		"vendor/k8s.io/apimachinery/pkg/runtime/schema/schema.go": `package schema
type GroupVersion struct{ Group, Version string }
type GroupVersionResource struct{ Group, Version, Resource string }
func (gv GroupVersion) WithResource(r string) GroupVersionResource {
	return GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: r}
}
`,
		"vendor/k8s.io/api/core/v1/types.go": `package v1
import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)
var SchemeGroupVersion = schema.GroupVersion{Version: "v1"}
type Pod struct {
	ObjectMeta metav1.ObjectMeta
	Spec       PodSpec
	Status     PodStatus
}
type PodSpec struct {
	Volumes               []Volume
	ActiveDeadlineSeconds *int64
	Hostname              string
}
type Volume struct {
	Name         string
	VolumeSource VolumeSource
}
type VolumeSource struct {
	EmptyDir              *EmptyDirVolumeSource
	PersistentVolumeClaim *PersistentVolumeClaimVolumeSource
}
type EmptyDirVolumeSource struct{ Medium string }
type PersistentVolumeClaimVolumeSource struct{ ClaimName string }
type PodStatus struct {
	Phase string
}
`,
		"vendor/k8s.io/client-go/listers/core/v1/pod.go": `package v1
import corev1 "k8s.io/api/core/v1"
type PodLister interface {
	List() ([]*corev1.Pod, error)
}
`,
		"vendor/k8s.io/client-go/informers/core/v1/pod.go": `package v1
import listers "k8s.io/client-go/listers/core/v1"
type PodInformer interface {
	Lister() listers.PodLister
}
type Interface interface {
	Pods() PodInformer
}
`,
		"vendor/k8s.io/client-go/informers/factory.go": `package informers
import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	coreinformers "k8s.io/client-go/informers/core/v1"
)
type GenericInformer interface{}
type SharedInformerFactory interface {
	Core() CoreInformers
	ForResource(resource schema.GroupVersionResource) (GenericInformer, error)
}
type CoreInformers interface {
	V1() coreinformers.Interface
}
`,
		"vendor/k8s.io/client-go/kubernetes/typed/core/v1/pod.go": `package v1
import corev1 "k8s.io/api/core/v1"
type PodInterface interface {
	Create(pod *corev1.Pod) (*corev1.Pod, error)
	Update(pod *corev1.Pod) (*corev1.Pod, error)
}
`,
		"vendor/k8s.io/apiserver/pkg/quota/v1/types.go": `package v1
import "k8s.io/apimachinery/pkg/runtime/schema"
type ListerForResourceFunc func(schema.GroupVersionResource) (any, error)
`,
		"vendor/k8s.io/apiserver/pkg/quota/v1/generic/lister.go": `package generic
import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/client-go/informers"
)
type ListFuncByNamespace func(namespace string) ([]any, error)
func ListerFuncForResourceFunc(f func(schema.GroupVersionResource) (informers.GenericInformer, error)) quota.ListerForResourceFunc {
	return func(gvr schema.GroupVersionResource) (any, error) { return f(gvr) }
}
func ListResourceUsingListerFunc(f quota.ListerForResourceFunc, resource schema.GroupVersionResource) ListFuncByNamespace {
	return func(namespace string) ([]any, error) { return nil, nil }
}
`,
		"quota/v1/pods.go": `package v1
import (
	corev1 "k8s.io/api/core/v1"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
)
type Evaluator struct {
	listPodFunc generic.ListFuncByNamespace
	listSvcFunc generic.ListFuncByNamespace
}
func NewEvaluator(f quota.ListerForResourceFunc) *Evaluator {
	return &Evaluator{
		listPodFunc: generic.ListResourceUsingListerFunc(f, corev1.SchemeGroupVersion.WithResource("pods")),
		listSvcFunc: generic.ListResourceUsingListerFunc(f, corev1.SchemeGroupVersion.WithResource("services")),
	}
}
func (e *Evaluator) Evaluate() {
	items, _ := e.listPodFunc("default")
	for _, item := range items {
		pod := item.(*corev1.Pod)
		_ = pod.Spec.ActiveDeadlineSeconds
	}
	svcs, _ := e.listSvcFunc("default")
	for _, item := range svcs {
		// Guarded by WithResource("services"), so Hostname must not be marked observed for Pods.
		if pod, ok := item.(*corev1.Pod); ok {
			_ = pod.Spec.Hostname
		}
	}
}
`,
		"ctrl/stateful.go": `package ctrl
import (
	corev1 "k8s.io/api/core/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
)
type Controller struct {
	lister corelisters.PodLister
	client coreclient.PodInterface
}
func NewController(podInformer coreinformers.PodInformer, client coreclient.PodInterface) *Controller {
	return &Controller{lister: podInformer.Lister(), client: client}
}
func isCreated(pod *corev1.Pod) bool {
	return pod.Status.Phase != ""
}
func updateStorage(pod *corev1.Pod) {
	var vols []corev1.Volume
	for i := range pod.Spec.Volumes {
		if pod.Spec.Volumes[i].VolumeSource.PersistentVolumeClaim != nil {
			vols = append(vols, pod.Spec.Volumes[i])
		}
	}
	pod.Spec.Volumes = vols
}
func runAll(fn func(int) error) { _ = fn(0) }
func (c *Controller) Sync(unSafeUpdate bool) {
	pods, _ := c.lister.List()
	replicas := make([]*corev1.Pod, 1)
	for _, p := range pods {
		replicas[0] = p
	}
	if replicas[0] == nil {
		replicas[0] = &corev1.Pod{}
	}
	runAll(func(i int) error {
		if !isCreated(replicas[i]) {
			_, err := c.client.Create(replicas[i])
			return err
		}
		cp := &corev1.Pod{Spec: replicas[i].Spec}
		updateStorage(cp)
		return nil
	})
}
`,
		"cmd/app/main.go": `package main
import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	"k8s.io/client-go/informers"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/synthetic/ctrl"
	quotav1 "k8s.io/synthetic/quota/v1"
)
var _ = schema.GroupVersionResource{}
func main() {
	var factory informers.SharedInformerFactory
	var client coreclient.PodInterface
	listerForResource := generic.ListerFuncForResourceFunc(factory.ForResource)
	quotav1.NewEvaluator(listerForResource).Evaluate()
	ctrl.NewController(factory.Core().V1().Pods(), client).Sync(false)
` + extraAppBody + "\n}\n",
	}
	for rel, content := range files {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}
	return Config{
		KubeRoot:         dir,
		BinaryPkg:        "k8s.io/synthetic/cmd/app",
		TargetPkgName:    "app",
		GeneratorCmd:     "informertrim-test",
		InformerIface:    "k8s.io/client-go/informers/core/v1.Interface",
		InformerMethod:   "Pods",
		InformerTypes:    []string{"k8s.io/client-go/informers/core/v1.PodInformer", "k8s.io/client-go/listers/core/v1.PodLister"},
		WriteClientIface: "k8s.io/client-go/kubernetes/typed/core/v1.PodInterface",
		RootType:         "k8s.io/api/core/v1.Pod",
		GeneratedFile:    filepath.Join(dir, "zz_generated.pod_trim.go"),
	}
}

func TestInformerTrimSynthetic(t *testing.T) {
	t.Run("indirect ListerForResourceFunc and guarded Create", func(t *testing.T) {
		cfg := writeSyntheticModule(t, "")
		rep := &captureReporter{t: t}
		out := string(Generate(rep, cfg))
		if len(rep.errors) > 0 {
			t.Fatalf("unexpected errors: %v", rep.errors)
		}
		if !strings.Contains(out, "ActiveDeadlineSeconds: pod.Spec.ActiveDeadlineSeconds") {
			t.Errorf("expected ActiveDeadlineSeconds from ListResourceUsingListerFunc(pods) to be preserved, got:\n%s", out)
		}
		if strings.Contains(out, "Hostname:") {
			t.Errorf("expected Hostname from non-pod WithResource(\"services\") lister to be trimmed, got:\n%s", out)
		}
		if strings.Contains(out, "EmptyDir:") || strings.Contains(out, "SelfLink:") {
			t.Errorf("expected unread VolumeSource.EmptyDir and ObjectMeta.SelfLink to remain trimmed after store-only slice write, got:\n%s", out)
		}
	})

	t.Run("rejects unguarded ForResource", func(t *testing.T) {
		cfg := writeSyntheticModule(t, `_, _ = factory.ForResource(schema.GroupVersionResource{Version: "v1", Resource: "pods"})`)
		rep := &captureReporter{t: t}
		_ = Generate(rep, cfg)
		if len(rep.errors) == 0 || !strings.Contains(strings.Join(rep.errors, "\n"), "unapproved SharedInformerFactory.ForResource") {
			t.Fatalf("expected unapproved ForResource error, got: %v", rep.errors)
		}
	})

	t.Run("rejects tainted Pod passed to Update via captured slice", func(t *testing.T) {
		cfg := writeSyntheticModule(t, `
	pods, _ := factory.Core().V1().Pods().Lister().List()
	slots := make([]*corev1.Pod, 1)
	for _, p := range pods {
		slots[0] = p
	}
	func() { _, _ = client.Update(slots[0]) }()
`)
		// Add corev1 import to cmd/app/main.go for this subtest.
		mainPath := filepath.Join(cfg.KubeRoot, "cmd", "app", "main.go")
		raw, err := os.ReadFile(mainPath)
		if err != nil {
			t.Fatal(err)
		}
		updated := strings.Replace(string(raw), "import (\n", "import (\n\tcorev1 \"k8s.io/api/core/v1\"\n", 1)
		if err := os.WriteFile(mainPath, []byte(updated), 0644); err != nil {
			t.Fatal(err)
		}
		rep := &captureReporter{t: t}
		_ = Generate(rep, cfg)
		if len(rep.errors) == 0 || !strings.Contains(strings.Join(rep.errors, "\n"), "tainted Pod from informer cache passed to") {
			t.Fatalf("expected tainted Update error, got: %v", rep.errors)
		}
	})
}
