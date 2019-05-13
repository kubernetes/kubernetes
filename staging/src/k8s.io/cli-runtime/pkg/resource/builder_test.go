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

package resource

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"sigs.k8s.io/yaml"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/meta/testrestmapper"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	restclientwatch "k8s.io/client-go/rest/watch"
	"k8s.io/client-go/restmapper"
	utiltesting "k8s.io/client-go/util/testing"

	// TODO we need to remove this linkage and create our own scheme
	"k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/scheme"
)

var (
	corev1GV     = schema.GroupVersion{Version: "v1"}
	corev1Codec  = scheme.Codecs.CodecForVersions(scheme.Codecs.LegacyCodec(corev1GV), scheme.Codecs.UniversalDecoder(corev1GV), corev1GV, corev1GV)
	metaAccessor = meta.NewAccessor()
)

func stringBody(body string) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(body)))
}

func watchBody(events ...watch.Event) string {
	buf := &bytes.Buffer{}
	codec := corev1Codec
	enc := restclientwatch.NewEncoder(streaming.NewEncoder(buf, codec), codec)
	for _, e := range events {
		enc.Encode(&e)
	}
	return buf.String()
}

func fakeClient() FakeClientFunc {
	return func(version schema.GroupVersion) (RESTClient, error) {
		return &fake.RESTClient{}, nil
	}
}

func fakeClientWith(testName string, t *testing.T, data map[string]string) FakeClientFunc {
	return func(version schema.GroupVersion) (RESTClient, error) {
		return &fake.RESTClient{
			GroupVersion:         corev1GV,
			NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				p := req.URL.Path
				q := req.URL.RawQuery
				if len(q) != 0 {
					p = p + "?" + q
				}
				body, ok := data[p]
				if !ok {
					t.Fatalf("%s: unexpected request: %s (%s)\n%#v", testName, p, req.URL, req)
				}
				header := http.Header{}
				header.Set("Content-Type", runtime.ContentTypeJSON)
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     header,
					Body:       stringBody(body),
				}, nil
			}),
		}, nil
	}
}

func testData() (*v1.PodList, *v1.ServiceList) {
	pods := &v1.PodList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "15",
		},
		Items: []v1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
				Spec:       V1DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "11"},
				Spec:       V1DeepEqualSafePodSpec(),
			},
		},
	}
	svc := &v1.ServiceList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "16",
		},
		Items: []v1.Service{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: v1.ServiceSpec{
					Type:            "ClusterIP",
					SessionAffinity: "None",
				},
			},
		},
	}
	return pods, svc
}

func streamTestData() (io.Reader, *v1.PodList, *v1.ServiceList) {
	pods, svc := testData()
	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write([]byte(runtime.EncodeOrDie(corev1Codec, pods)))
		w.Write([]byte(runtime.EncodeOrDie(corev1Codec, svc)))
	}()
	return r, pods, svc
}

func JSONToYAMLOrDie(in []byte) []byte {
	data, err := yaml.JSONToYAML(in)
	if err != nil {
		panic(err)
	}
	return data
}

func streamYAMLTestData() (io.Reader, *v1.PodList, *v1.ServiceList) {
	pods, svc := testData()
	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write(JSONToYAMLOrDie([]byte(runtime.EncodeOrDie(corev1Codec, pods))))
		w.Write([]byte("\n---\n"))
		w.Write(JSONToYAMLOrDie([]byte(runtime.EncodeOrDie(corev1Codec, svc))))
	}()
	return r, pods, svc
}

func streamTestObject(obj runtime.Object) io.Reader {
	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write([]byte(runtime.EncodeOrDie(corev1Codec, obj)))
	}()
	return r
}

type testVisitor struct {
	InjectErr error
	Infos     []*Info
}

func (v *testVisitor) Handle(info *Info, err error) error {
	if err != nil {
		return err
	}
	v.Infos = append(v.Infos, info)
	return v.InjectErr
}

func (v *testVisitor) Objects() []runtime.Object {
	objects := []runtime.Object{}
	for i := range v.Infos {
		objects = append(objects, v.Infos[i].Object)
	}
	return objects
}

var aPod string = `
{
    "kind": "Pod",
		"apiVersion": "` + corev1GV.String() + `",
    "metadata": {
        "name": "busybox{id}",
        "labels": {
            "name": "busybox{id}"
        }
    },
    "spec": {
        "containers": [
            {
                "name": "busybox",
                "image": "busybox",
                "command": [
                    "sleep",
                    "3600"
                ],
                "imagePullPolicy": "IfNotPresent"
            }
        ],
        "restartPolicy": "Always"
    }
}
`

var aRC string = `
{
    "kind": "ReplicationController",
		"apiVersion": "` + corev1GV.String() + `",
    "metadata": {
        "name": "busybox{id}",
        "labels": {
            "app": "busybox"
        }
    },
    "spec": {
        "replicas": 1,
        "template": {
            "metadata": {
                "name": "busybox{id}",
                "labels": {
                    "app": "busybox{id}"
                }
            },
            "spec": {
                "containers": [
                    {
                        "name": "busybox",
                        "image": "busybox",
                        "command": [
                            "sleep",
                            "3600"
                        ],
                        "imagePullPolicy": "IfNotPresent"
                    }
                ],
                "restartPolicy": "Always"
            }
        }
    }
}
`

func newDefaultBuilder() *Builder {
	return newDefaultBuilderWith(fakeClient())
}

func newDefaultBuilderWith(fakeClientFn FakeClientFunc) *Builder {
	return NewFakeBuilder(
		fakeClientFn,
		func() (meta.RESTMapper, error) {
			return testrestmapper.TestOnlyStaticRESTMapper(scheme.Scheme), nil
		},
		func() (restmapper.CategoryExpander, error) {
			return FakeCategoryExpander, nil
		}).
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...)
}

type errorRestMapper struct {
	meta.RESTMapper
	err error
}

func (l *errorRestMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	return nil, l.err
}

func newDefaultBuilderWithMapperError(fakeClientFn FakeClientFunc, err error) *Builder {
	return NewFakeBuilder(
		fakeClientFn,
		func() (meta.RESTMapper, error) {
			return &errorRestMapper{
				RESTMapper: testrestmapper.TestOnlyStaticRESTMapper(scheme.Scheme),
				err:        err,
			}, nil
		},
		func() (restmapper.CategoryExpander, error) {
			return FakeCategoryExpander, nil
		}).
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...)
}

func TestPathBuilderAndVersionedObjectNotDefaulted(t *testing.T) {
	b := newDefaultBuilder().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{"../../artifacts/kitten-rc.yaml"}})

	test := &testVisitor{}
	singleItemImplied := false

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || !singleItemImplied || len(test.Infos) != 1 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}

	info := test.Infos[0]
	if info.Name != "update-demo-kitten" || info.Namespace != "" || info.Object == nil {
		t.Errorf("unexpected info: %#v", info)
	}

	obj := info.Object
	version, ok := obj.(*v1.ReplicationController)
	// versioned object does not have defaulting applied
	if obj == nil || !ok || version.Spec.Replicas != nil {
		t.Errorf("unexpected versioned object: %#v", obj)
	}
}

func TestNodeBuilder(t *testing.T) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node1", Namespace: "should-not-have", ResourceVersion: "10"},
		Spec:       v1.NodeSpec{},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1000m"),
				v1.ResourceMemory: resource.MustParse("1Mi"),
			},
		},
	}
	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write([]byte(runtime.EncodeOrDie(corev1Codec, node)))
	}()

	b := newDefaultBuilder().NamespaceParam("test").Stream(r, "STDIN")

	test := &testVisitor{}

	err := b.Do().Visit(test.Handle)
	if err != nil || len(test.Infos) != 1 {
		t.Fatalf("unexpected response: %v %#v", err, test.Infos)
	}
	info := test.Infos[0]
	if info.Name != "node1" || info.Namespace != "" || info.Object == nil {
		t.Errorf("unexpected info: %#v", info)
	}
}

func createTestDir(t *testing.T, path string) {
	if err := os.MkdirAll(path, 0750); err != nil {
		t.Fatalf("error creating test dir: %v", err)
	}
}

func writeTestFile(t *testing.T, path string, contents string) {
	if err := ioutil.WriteFile(path, []byte(contents), 0644); err != nil {
		t.Fatalf("error creating test file %#v", err)
	}
}

func TestFilenameOptionsValidate(t *testing.T) {
	testcases := []struct {
		filenames []string
		kustomize string
		recursive bool
		errExp    bool
		msgExp    string
	}{
		{
			filenames: []string{"file"},
			kustomize: "dir",
			errExp:    true,
			msgExp:    "only one of -f or -k can be specified",
		},
		{
			kustomize: "dir",
			recursive: true,
			errExp:    true,
			msgExp:    "the -k flag can't be used with -f or -R",
		},
		{
			filenames: []string{"file"},
			errExp:    false,
		},
		{
			filenames: []string{"dir"},
			recursive: true,
			errExp:    false,
		},
		{
			kustomize: "dir",
			errExp:    false,
		},
	}
	for _, testcase := range testcases {
		o := &FilenameOptions{
			Kustomize: testcase.kustomize,
			Filenames: testcase.filenames,
			Recursive: testcase.recursive,
		}
		errs := o.validate()
		if testcase.errExp {
			if len(errs) == 0 {
				t.Fatalf("expected error not happened")
			}
			if errs[0].Error() != testcase.msgExp {
				t.Fatalf("expected %s, but got %#v", testcase.msgExp, errs[0])
			}
		} else {
			if len(errs) > 0 {
				t.Fatalf("Unexpected error %#v", errs)
			}
		}
	}
}

func TestPathBuilderWithMultiple(t *testing.T) {
	// create test dirs
	tmpDir, err := utiltesting.MkTmpdir("recursive_test_multiple")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	createTestDir(t, fmt.Sprintf("%s/%s", tmpDir, "recursive/pod/pod_1"))
	createTestDir(t, fmt.Sprintf("%s/%s", tmpDir, "recursive/rc/rc_1"))
	createTestDir(t, fmt.Sprintf("%s/%s", tmpDir, "inode/hardlink"))
	defer os.RemoveAll(tmpDir)

	// create test files
	writeTestFile(t, fmt.Sprintf("%s/recursive/pod/busybox.json", tmpDir), strings.Replace(aPod, "{id}", "0", -1))
	writeTestFile(t, fmt.Sprintf("%s/recursive/pod/pod_1/busybox.json", tmpDir), strings.Replace(aPod, "{id}", "1", -1))
	writeTestFile(t, fmt.Sprintf("%s/recursive/rc/busybox.json", tmpDir), strings.Replace(aRC, "{id}", "0", -1))
	writeTestFile(t, fmt.Sprintf("%s/recursive/rc/rc_1/busybox.json", tmpDir), strings.Replace(aRC, "{id}", "1", -1))
	writeTestFile(t, fmt.Sprintf("%s/inode/hardlink/busybox.json", tmpDir), strings.Replace(aPod, "{id}", "0", -1))
	if err := os.Link(fmt.Sprintf("%s/inode/hardlink/busybox.json", tmpDir), fmt.Sprintf("%s/inode/hardlink/busybox-link.json", tmpDir)); err != nil {
		t.Fatalf("error creating test file: %v", err)
	}

	tests := []struct {
		name          string
		object        runtime.Object
		recursive     bool
		directory     string
		expectedNames []string
	}{
		{"pod", &v1.Pod{}, false, "../../artifacts/pod.yaml", []string{"nginx"}},
		{"recursive-pod", &v1.Pod{}, true, fmt.Sprintf("%s/recursive/pod", tmpDir), []string{"busybox0", "busybox1"}},
		{"rc", &v1.ReplicationController{}, false, "../../artifacts/redis-master-controller.yaml", []string{"redis-master"}},
		{"recursive-rc", &v1.ReplicationController{}, true, fmt.Sprintf("%s/recursive/rc", tmpDir), []string{"busybox0", "busybox1"}},
		{"hardlink", &v1.Pod{}, false, fmt.Sprintf("%s/inode/hardlink/busybox-link.json", tmpDir), []string{"busybox0"}},
		{"hardlink", &v1.Pod{}, true, fmt.Sprintf("%s/inode/hardlink/busybox-link.json", tmpDir), []string{"busybox0"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newDefaultBuilder().
				FilenameParam(false, &FilenameOptions{Recursive: tt.recursive, Filenames: []string{tt.directory}}).
				NamespaceParam("test").DefaultNamespace()

			testVisitor := &testVisitor{}
			singleItemImplied := false

			err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(testVisitor.Handle)
			if err != nil {
				t.Fatalf("unexpected response: %v %t %#v %s", err, singleItemImplied, testVisitor.Infos, tt.name)
			}

			info := testVisitor.Infos

			for i, v := range info {
				switch tt.object.(type) {
				case *v1.Pod:
					if _, ok := v.Object.(*v1.Pod); !ok || v.Name != tt.expectedNames[i] || v.Namespace != "test" {
						t.Errorf("unexpected info: %v", spew.Sdump(v.Object))
					}
				case *v1.ReplicationController:
					if _, ok := v.Object.(*v1.ReplicationController); !ok || v.Name != tt.expectedNames[i] || v.Namespace != "test" {
						t.Errorf("unexpected info: %v", spew.Sdump(v.Object))
					}
				}
			}
		})
	}
}

func TestPathBuilderWithMultipleInvalid(t *testing.T) {
	// create test dirs
	tmpDir, err := utiltesting.MkTmpdir("recursive_test_multiple_invalid")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	createTestDir(t, fmt.Sprintf("%s/%s", tmpDir, "inode/symlink/pod"))
	defer os.RemoveAll(tmpDir)

	// create test files
	writeTestFile(t, fmt.Sprintf("%s/inode/symlink/pod/busybox.json", tmpDir), strings.Replace(aPod, "{id}", "0", -1))
	if err := os.Symlink(fmt.Sprintf("%s/inode/symlink/pod", tmpDir), fmt.Sprintf("%s/inode/symlink/pod-link", tmpDir)); err != nil {
		t.Fatalf("error creating test file: %v", err)
	}
	if err := os.Symlink(fmt.Sprintf("%s/inode/symlink/loop", tmpDir), fmt.Sprintf("%s/inode/symlink/loop", tmpDir)); err != nil {
		t.Fatalf("error creating test file: %v", err)
	}

	tests := []struct {
		name      string
		recursive bool
		directory string
	}{
		{"symlink", false, fmt.Sprintf("%s/inode/symlink/pod-link", tmpDir)},
		{"symlink", true, fmt.Sprintf("%s/inode/symlink/pod-link", tmpDir)},
		{"loop", false, fmt.Sprintf("%s/inode/symlink/loop", tmpDir)},
		{"loop", true, fmt.Sprintf("%s/inode/symlink/loop", tmpDir)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newDefaultBuilder().
				FilenameParam(false, &FilenameOptions{Recursive: tt.recursive, Filenames: []string{tt.directory}}).
				NamespaceParam("test").DefaultNamespace()

			testVisitor := &testVisitor{}
			singleItemImplied := false

			err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(testVisitor.Handle)
			if err == nil {
				t.Fatalf("unexpected response: %v %t %#v %s", err, singleItemImplied, testVisitor.Infos, tt.name)
			}
		})
	}
}

func TestDirectoryBuilder(t *testing.T) {
	b := newDefaultBuilder().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{"../../artifacts/guestbook"}}).
		NamespaceParam("test").DefaultNamespace()

	test := &testVisitor{}
	singleItemImplied := false

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || singleItemImplied || len(test.Infos) < 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}

	found := false
	for _, info := range test.Infos {
		if info.Name == "redis-master" && info.Namespace == "test" && info.Object != nil {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("unexpected responses: %#v", test.Infos)
	}
}

func setupKustomizeDirectory() (string, error) {
	path, err := ioutil.TempDir("/tmp", "")
	if err != nil {
		return "", err
	}

	contents := map[string]string{
		"configmap.yaml": `
apiVersion: v1
kind: ConfigMap
metadata:
  name: the-map
data:
  altGreeting: "Good Morning!"
  enableRisky: "false"
`,
		"deployment.yaml": `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: the-deployment
spec:
  replicas: 3
  template:
    metadata:
      labels:
        deployment: hello
    spec:
      containers:
      - name: the-container
        image: monopole/hello:1
        command: ["/hello",
                  "--port=8080",
                  "--enableRiskyFeature=$(ENABLE_RISKY)"]
        ports:
        - containerPort: 8080
        env:
        - name: ALT_GREETING
          valueFrom:
            configMapKeyRef:
              name: the-map
              key: altGreeting
        - name: ENABLE_RISKY
          valueFrom:
            configMapKeyRef:
              name: the-map
              key: enableRisky
`,
		"service.yaml": `
kind: Service
apiVersion: v1
metadata:
  name: the-service
spec:
  selector:
    deployment: hello
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 8666
    targetPort: 8080
`,
		"kustomization.yaml": `
nameprefix: test-
resources:
- deployment.yaml
- service.yaml
- configmap.yaml
`,
	}

	for filename, content := range contents {
		err = ioutil.WriteFile(filepath.Join(path, filename), []byte(content), 0660)
		if err != nil {
			return "", err
		}
	}
	return path, nil
}

func TestKustomizeDirectoryBuilder(t *testing.T) {
	dir, err := setupKustomizeDirectory()
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	defer os.RemoveAll(dir)

	tests := []struct {
		directory     string
		expectErr     bool
		errMsg        string
		number        int
		expectedNames []string
	}{
		{
			directory: "../../artifacts/guestbook",
			expectErr: true,
			errMsg:    "unable to find one of 'kustomization.yaml', 'kustomization.yml' or 'Kustomization'",
		},
		{
			directory:     dir,
			expectErr:     false,
			expectedNames: []string{"test-the-map", "test-the-deployment", "test-the-service"},
		},
		{
			directory: filepath.Join(dir, "kustomization.yaml"),
			expectErr: true,
			errMsg:    "must be a directory to be a root",
		},
		{
			directory: "../../artifacts/kustomization/should-not-load.yaml",
			expectErr: true,
			errMsg:    "must be a directory to be a root",
		},
	}
	for _, tt := range tests {
		b := newDefaultBuilder().
			FilenameParam(false, &FilenameOptions{Kustomize: tt.directory}).
			NamespaceParam("test").DefaultNamespace()
		test := &testVisitor{}
		err := b.Do().Visit(test.Handle)
		if tt.expectErr {
			if err == nil {
				t.Fatalf("expected error unhappened")
			}
			if !strings.Contains(err.Error(), tt.errMsg) {
				t.Fatalf("expected %s but got %s", tt.errMsg, err.Error())
			}
		} else {
			if err != nil || len(test.Infos) < tt.number {
				t.Fatalf("unexpected response: %v %#v", err, test.Infos)
			}
			contained := func(name string) bool {
				for _, info := range test.Infos {
					if info.Name == name && info.Namespace == "test" && info.Object != nil {
						return true
					}
				}
				return false
			}

			allFound := true
			for _, name := range tt.expectedNames {
				if !contained(name) {
					allFound = false
				}
			}
			if !allFound {
				t.Errorf("unexpected responses: %#v", test.Infos)
			}
		}
	}
}

func TestNamespaceOverride(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(runtime.EncodeOrDie(corev1Codec, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "test"}})))
	}))
	defer s.Close()

	b := newDefaultBuilder().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{s.URL}}).
		NamespaceParam("test")

	test := &testVisitor{}

	err := b.Do().Visit(test.Handle)
	if err != nil || len(test.Infos) != 1 && test.Infos[0].Namespace != "foo" {
		t.Fatalf("unexpected response: %v %#v", err, test.Infos)
	}

	b = newDefaultBuilder().
		FilenameParam(true, &FilenameOptions{Recursive: false, Filenames: []string{s.URL}}).
		NamespaceParam("test")

	test = &testVisitor{}

	err = b.Do().Visit(test.Handle)
	if err == nil {
		t.Fatalf("expected namespace error. got: %#v", test.Infos)
	}
}

func TestURLBuilder(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(runtime.EncodeOrDie(corev1Codec, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "test"}})))
		w.Write([]byte(runtime.EncodeOrDie(corev1Codec, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "test1"}})))
	}))
	defer s.Close()

	b := newDefaultBuilder().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{s.URL}}).
		NamespaceParam("foo")

	test := &testVisitor{}

	err := b.Do().Visit(test.Handle)
	if err != nil || len(test.Infos) != 2 {
		t.Fatalf("unexpected response: %v %#v", err, test.Infos)
	}
	info := test.Infos[0]
	if info.Name != "test" || info.Namespace != "foo" || info.Object == nil {
		t.Errorf("unexpected info: %#v", info)
	}

	info = test.Infos[1]
	if info.Name != "test1" || info.Namespace != "foo" || info.Object == nil {
		t.Errorf("unexpected info: %#v", info)
	}

}

func TestURLBuilderRequireNamespace(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(runtime.EncodeOrDie(corev1Codec, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "test"}})))
	}))
	defer s.Close()

	b := newDefaultBuilder().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{s.URL}}).
		NamespaceParam("test").RequireNamespace()

	test := &testVisitor{}
	singleItemImplied := false

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err == nil || !singleItemImplied || len(test.Infos) != 0 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}
}

func TestReplaceAliases(t *testing.T) {
	tests := []struct {
		name     string
		arg      string
		expected string
	}{
		{
			name:     "no-replacement",
			arg:      "service",
			expected: "service",
		},
		{
			name:     "all-replacement",
			arg:      "all",
			expected: "pods,replicationcontrollers,services,statefulsets.apps,horizontalpodautoscalers.autoscaling,jobs.batch,cronjobs.batch,daemonsets.extensions,deployments.extensions,replicasets.extensions",
		},
		{
			name:     "alias-in-comma-separated-arg",
			arg:      "all,secrets",
			expected: "pods,replicationcontrollers,services,statefulsets.apps,horizontalpodautoscalers.autoscaling,jobs.batch,cronjobs.batch,daemonsets.extensions,deployments.extensions,replicasets.extensions,secrets",
		},
	}

	b := newDefaultBuilder()

	for _, test := range tests {
		replaced := b.ReplaceAliases(test.arg)
		if replaced != test.expected {
			t.Errorf("%s: unexpected argument: expected %s, got %s", test.name, test.expected, replaced)
		}
	}
}

func TestResourceByName(t *testing.T) {
	pods, _ := testData()
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo": runtime.EncodeOrDie(corev1Codec, &pods.Items[0]),
	})).NamespaceParam("test")

	test := &testVisitor{}
	singleItemImplied := false

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceTypeOrNameArgs(true, "pods", "foo")

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || !singleItemImplied || len(test.Infos) != 1 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}
	if !apiequality.Semantic.DeepEqual(&pods.Items[0], test.Objects()[0]) {
		t.Errorf("unexpected object: %#v", test.Objects()[0])
	}

	mapping, err := b.Do().ResourceMapping()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != (schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}) {
		t.Errorf("unexpected resource mapping: %#v", mapping)
	}
}

func TestRestMappingErrors(t *testing.T) {
	pods, _ := testData()
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo": runtime.EncodeOrDie(corev1Codec, &pods.Items[0]),
	})).NamespaceParam("test")

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	test := &testVisitor{}
	singleItemImplied := false

	b.ResourceTypeOrNameArgs(true, "foo", "bar")

	// ensure that requesting a resource we _know_ not to exist results in an expected *meta.NoKindMatchError
	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil {
		if !strings.Contains(err.Error(), "server doesn't have a resource type \"foo\"") {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	expectedErr := fmt.Errorf("expected error")
	b = newDefaultBuilderWithMapperError(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo": runtime.EncodeOrDie(corev1Codec, &pods.Items[0]),
	}), expectedErr).NamespaceParam("test")

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	// ensure we request a resource we know not to exist. This way, we
	// end up taking the codepath we want to test in the resource builder
	b.ResourceTypeOrNameArgs(true, "foo", "bar")

	// ensure that receiving an error for any reason other than a non-existent resource is returned as-is
	err = b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil && !strings.Contains(err.Error(), expectedErr.Error()) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestMultipleResourceByTheSameName(t *testing.T) {
	pods, svcs := testData()
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo":     runtime.EncodeOrDie(corev1Codec, &pods.Items[0]),
		"/namespaces/test/pods/baz":     runtime.EncodeOrDie(corev1Codec, &pods.Items[1]),
		"/namespaces/test/services/foo": runtime.EncodeOrDie(corev1Codec, &svcs.Items[0]),
		"/namespaces/test/services/baz": runtime.EncodeOrDie(corev1Codec, &svcs.Items[0]),
	})).
		NamespaceParam("test")

	test := &testVisitor{}
	singleItemImplied := false

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceTypeOrNameArgs(true, "pods,services", "foo", "baz")

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || singleItemImplied || len(test.Infos) != 4 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}
	if !apiequality.Semantic.DeepDerivative([]runtime.Object{&pods.Items[0], &pods.Items[1], &svcs.Items[0], &svcs.Items[0]}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}

	if _, err := b.Do().ResourceMapping(); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestRequestModifier(t *testing.T) {
	for _, tc := range []struct {
		name string
		f    func(t *testing.T, got **rest.Request) *Builder
	}{
		{
			name: "simple",
			f: func(t *testing.T, got **rest.Request) *Builder {
				return newDefaultBuilderWith(fakeClientWith(t.Name(), t, nil)).
					NamespaceParam("foo").
					TransformRequests(func(req *rest.Request) {
						*got = req
					}).
					ResourceNames("", "services/baz").
					RequireObject(false)
			},
		},
		{
			name: "flatten",
			f: func(t *testing.T, got **rest.Request) *Builder {
				pods, _ := testData()
				return newDefaultBuilderWith(fakeClientWith(t.Name(), t, map[string]string{
					"/namespaces/foo/pods": runtime.EncodeOrDie(corev1Codec, pods),
				})).
					NamespaceParam("foo").
					TransformRequests(func(req *rest.Request) {
						*got = req
					}).
					ResourceTypeOrNameArgs(true, "pods").
					Flatten()
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got *rest.Request
			b := tc.f(t, &got)
			i, err := b.Do().Infos()
			if err != nil {
				t.Fatal(err)
			}
			req := i[0].Client.Get()
			if got != req {
				t.Fatalf("request was not received by modifier: %#v", req)
			}
		})
	}
}

func TestResourceNames(t *testing.T) {
	pods, svc := testData()
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo":     runtime.EncodeOrDie(corev1Codec, &pods.Items[0]),
		"/namespaces/test/services/baz": runtime.EncodeOrDie(corev1Codec, &svc.Items[0]),
	})).NamespaceParam("test")

	test := &testVisitor{}

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceNames("pods", "foo", "services/baz")

	err := b.Do().Visit(test.Handle)
	if err != nil || len(test.Infos) != 2 {
		t.Fatalf("unexpected response: %v %#v", err, test.Infos)
	}
	if !apiequality.Semantic.DeepEqual(&pods.Items[0], test.Objects()[0]) {
		t.Errorf("unexpected object: \n%#v, expected: \n%#v", test.Objects()[0], &pods.Items[0])
	}
	if !apiequality.Semantic.DeepEqual(&svc.Items[0], test.Objects()[1]) {
		t.Errorf("unexpected object: \n%#v, expected: \n%#v", test.Objects()[1], &svc.Items[0])
	}
}

func TestResourceNamesWithoutResource(t *testing.T) {
	pods, svc := testData()
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo":     runtime.EncodeOrDie(corev1Codec, &pods.Items[0]),
		"/namespaces/test/services/baz": runtime.EncodeOrDie(corev1Codec, &svc.Items[0]),
	})).NamespaceParam("test")

	test := &testVisitor{}

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceNames("", "foo", "services/baz")

	err := b.Do().Visit(test.Handle)
	if err == nil || !strings.Contains(err.Error(), "must be RESOURCE/NAME") {
		t.Fatalf("unexpected response: %v", err)
	}
}

func TestResourceByNameWithoutRequireObject(t *testing.T) {
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{})).NamespaceParam("test")

	test := &testVisitor{}
	singleItemImplied := false

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceTypeOrNameArgs(true, "pods", "foo").RequireObject(false)

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || !singleItemImplied || len(test.Infos) != 1 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}
	if test.Infos[0].Name != "foo" {
		t.Errorf("unexpected name: %#v", test.Infos[0].Name)
	}
	if test.Infos[0].Object != nil {
		t.Errorf("unexpected object: %#v", test.Infos[0].Object)
	}

	mapping, err := b.Do().ResourceMapping()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.GroupVersionKind.Kind != "Pod" || mapping.Resource != (schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}) {
		t.Errorf("unexpected resource mapping: %#v", mapping)
	}
}

func TestResourceByNameAndEmptySelector(t *testing.T) {
	pods, _ := testData()
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo": runtime.EncodeOrDie(corev1Codec, &pods.Items[0]),
	})).
		NamespaceParam("test").
		LabelSelectorParam("").
		ResourceTypeOrNameArgs(true, "pods", "foo")

	singleItemImplied := false
	infos, err := b.Do().IntoSingleItemImplied(&singleItemImplied).Infos()
	if err != nil || !singleItemImplied || len(infos) != 1 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, infos)
	}
	if !apiequality.Semantic.DeepEqual(&pods.Items[0], infos[0].Object) {
		t.Errorf("unexpected object: %#v", infos[0])
	}

	mapping, err := b.Do().ResourceMapping()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != (schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}) {
		t.Errorf("unexpected resource mapping: %#v", mapping)
	}
}

func TestLabelSelector(t *testing.T) {
	pods, svc := testData()
	labelKey := metav1.LabelSelectorQueryParam(corev1GV.String())
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods?" + labelKey + "=a%3Db":     runtime.EncodeOrDie(corev1Codec, pods),
		"/namespaces/test/services?" + labelKey + "=a%3Db": runtime.EncodeOrDie(corev1Codec, svc),
	})).
		LabelSelectorParam("a=b").
		NamespaceParam("test").
		Flatten()

	test := &testVisitor{}
	singleItemImplied := false

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceTypeOrNameArgs(true, "pods,service")

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || singleItemImplied || len(test.Infos) != 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}
	if !apiequality.Semantic.DeepDerivative([]runtime.Object{&pods.Items[0], &pods.Items[1], &svc.Items[0]}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}

	if _, err := b.Do().ResourceMapping(); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestLabelSelectorRequiresKnownTypes(t *testing.T) {
	b := newDefaultBuilder().
		LabelSelectorParam("a=b").
		NamespaceParam("test").
		ResourceTypes("unknown")

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestFieldSelector(t *testing.T) {
	pods, svc := testData()
	fieldKey := metav1.FieldSelectorQueryParam(corev1GV.String())
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods?" + fieldKey + "=a%3Db":     runtime.EncodeOrDie(corev1Codec, pods),
		"/namespaces/test/services?" + fieldKey + "=a%3Db": runtime.EncodeOrDie(corev1Codec, svc),
	})).
		FieldSelectorParam("a=b").
		NamespaceParam("test").
		Flatten()

	test := &testVisitor{}
	singleItemImplied := false

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceTypeOrNameArgs(true, "pods,service")

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || singleItemImplied || len(test.Infos) != 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}
	if !apiequality.Semantic.DeepDerivative([]runtime.Object{&pods.Items[0], &pods.Items[1], &svc.Items[0]}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}

	if _, err := b.Do().ResourceMapping(); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestFieldSelectorRequiresKnownTypes(t *testing.T) {
	b := newDefaultBuilder().
		FieldSelectorParam("a=b").
		NamespaceParam("test").
		ResourceTypes("unknown")

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestSingleResourceType(t *testing.T) {
	b := newDefaultBuilder().
		LabelSelectorParam("a=b").
		SingleResourceType().
		ResourceTypeOrNameArgs(true, "pods,services")

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestResourceTuple(t *testing.T) {
	expectNoErr := func(err error) bool { return err == nil }
	expectErr := func(err error) bool { return err != nil }
	testCases := map[string]struct {
		args  []string
		errFn func(error) bool
	}{
		"valid": {
			args:  []string{"pods/foo"},
			errFn: expectNoErr,
		},
		"valid multiple with name indirection": {
			args:  []string{"pods/foo", "pod/bar"},
			errFn: expectNoErr,
		},
		"valid multiple with namespaced and non-namespaced types": {
			args:  []string{"nodes/foo", "pod/bar"},
			errFn: expectNoErr,
		},
		"mixed arg types": {
			args:  []string{"pods/foo", "bar"},
			errFn: expectErr,
		},
		/*"missing resource": {
			args:  []string{"pods/foo2"},
			errFn: expectNoErr, // not an error because resources are lazily visited
		},*/
		"comma in resource": {
			args:  []string{",pods/foo"},
			errFn: expectErr,
		},
		"multiple types in resource": {
			args:  []string{"pods,services/foo"},
			errFn: expectErr,
		},
		"unknown resource type": {
			args:  []string{"unknown/foo"},
			errFn: expectErr,
		},
		"leading slash": {
			args:  []string{"/bar"},
			errFn: expectErr,
		},
		"trailing slash": {
			args:  []string{"bar/"},
			errFn: expectErr,
		},
	}
	for k, tt := range testCases {
		t.Run("using default namespace", func(t *testing.T) {
			for _, requireObject := range []bool{true, false} {
				expectedRequests := map[string]string{}
				if requireObject {
					pods, _ := testData()
					expectedRequests = map[string]string{
						"/namespaces/test/pods/foo": runtime.EncodeOrDie(corev1Codec, &pods.Items[0]),
						"/namespaces/test/pods/bar": runtime.EncodeOrDie(corev1Codec, &pods.Items[0]),
						"/nodes/foo":                runtime.EncodeOrDie(corev1Codec, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}),
					}
				}
				b := newDefaultBuilderWith(fakeClientWith(k, t, expectedRequests)).
					NamespaceParam("test").DefaultNamespace().
					ResourceTypeOrNameArgs(true, tt.args...).RequireObject(requireObject)

				r := b.Do()

				if !tt.errFn(r.Err()) {
					t.Errorf("%s: unexpected error: %v", k, r.Err())
				}
				if r.Err() != nil {
					continue
				}
				switch {
				case (r.singleItemImplied && len(tt.args) != 1),
					(!r.singleItemImplied && len(tt.args) == 1):
					t.Errorf("%s: result had unexpected singleItemImplied value", k)
				}
				info, err := r.Infos()
				if err != nil {
					// test error
					continue
				}
				if len(info) != len(tt.args) {
					t.Errorf("%s: unexpected number of infos returned: %#v", k, info)
				}
			}
		})
	}
}

func TestStream(t *testing.T) {
	r, pods, rc := streamTestData()
	b := newDefaultBuilder().
		NamespaceParam("test").Stream(r, "STDIN").Flatten()

	test := &testVisitor{}
	singleItemImplied := false

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || singleItemImplied || len(test.Infos) != 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}
	if !apiequality.Semantic.DeepDerivative([]runtime.Object{&pods.Items[0], &pods.Items[1], &rc.Items[0]}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}
}

func TestYAMLStream(t *testing.T) {
	r, pods, rc := streamYAMLTestData()
	b := newDefaultBuilder().
		NamespaceParam("test").Stream(r, "STDIN").Flatten()

	test := &testVisitor{}
	singleItemImplied := false

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || singleItemImplied || len(test.Infos) != 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}
	if !apiequality.Semantic.DeepDerivative([]runtime.Object{&pods.Items[0], &pods.Items[1], &rc.Items[0]}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}
}

func TestMultipleObject(t *testing.T) {
	r, pods, svc := streamTestData()
	obj, err := newDefaultBuilder().
		NamespaceParam("test").Stream(r, "STDIN").Flatten().
		Do().Object()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := &v1.List{
		Items: []runtime.RawExtension{
			{Object: &pods.Items[0]},
			{Object: &pods.Items[1]},
			{Object: &svc.Items[0]},
		},
	}
	if !apiequality.Semantic.DeepDerivative(expected, obj) {
		t.Errorf("unexpected visited objects: %#v", obj)
	}
}

func TestContinueOnErrorVisitor(t *testing.T) {
	r, _, _ := streamTestData()
	req := newDefaultBuilder().
		ContinueOnError().
		NamespaceParam("test").Stream(r, "STDIN").Flatten().
		Do()
	count := 0
	testErr := fmt.Errorf("test error")
	err := req.Visit(func(_ *Info, _ error) error {
		count++
		if count > 1 {
			return testErr
		}
		return nil
	})
	if err == nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if count != 3 {
		t.Fatalf("did not visit all infos: %d", count)
	}
	agg, ok := err.(utilerrors.Aggregate)
	if !ok {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(agg.Errors()) != 2 || agg.Errors()[0] != testErr || agg.Errors()[1] != testErr {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestSingleItemImpliedObject(t *testing.T) {
	obj, err := newDefaultBuilder().
		NamespaceParam("test").DefaultNamespace().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{"../../artifacts/guestbook/redis-master-controller.yaml"}}).
		Flatten().
		Do().Object()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rc, ok := obj.(*v1.ReplicationController)
	if !ok {
		t.Fatalf("unexpected object: %#v", obj)
	}
	if rc.Name != "redis-master" || rc.Namespace != "test" {
		t.Errorf("unexpected controller: %#v", rc)
	}
}

func TestSingleItemImpliedObjectNoExtension(t *testing.T) {
	obj, err := newDefaultBuilder().
		NamespaceParam("test").DefaultNamespace().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{"../../artifacts/pod.yaml"}}).
		Flatten().
		Do().Object()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pod, ok := obj.(*v1.Pod)
	if !ok {
		t.Fatalf("unexpected object: %#v", obj)
	}
	if pod.Name != "nginx" || pod.Namespace != "test" {
		t.Errorf("unexpected pod: %#v", pod)
	}
}

func TestSingleItemImpliedRootScopedObject(t *testing.T) {
	node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test"}}
	r := streamTestObject(node)
	infos, err := newDefaultBuilder().
		NamespaceParam("test").DefaultNamespace().
		Stream(r, "STDIN").
		Flatten().
		Do().Infos()

	if err != nil || len(infos) != 1 {
		t.Fatalf("unexpected error: %v", err)
	}

	if infos[0].Namespace != "" {
		t.Errorf("namespace should be empty: %#v", infos[0])
	}
	n, ok := infos[0].Object.(*v1.Node)
	if !ok {
		t.Fatalf("unexpected object: %#v", infos[0].Object)
	}
	if n.Name != "test" || n.Namespace != "" {
		t.Errorf("unexpected object: %#v", n)
	}
}

func TestListObject(t *testing.T) {
	pods, _ := testData()
	labelKey := metav1.LabelSelectorQueryParam(corev1GV.String())
	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods?" + labelKey + "=a%3Db": runtime.EncodeOrDie(corev1Codec, pods),
	})).
		LabelSelectorParam("a=b").
		NamespaceParam("test").
		ResourceTypeOrNameArgs(true, "pods").
		Flatten()

	obj, err := b.Do().Object()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	list, ok := obj.(*v1.List)
	if !ok {
		t.Fatalf("unexpected object: %#v", obj)
	}
	if list.ResourceVersion != pods.ResourceVersion || len(list.Items) != 2 {
		t.Errorf("unexpected list: %#v", list)
	}

	mapping, err := b.Do().ResourceMapping()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != (schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}) {
		t.Errorf("unexpected resource mapping: %#v", mapping)
	}
}

func TestListObjectWithDifferentVersions(t *testing.T) {
	pods, svc := testData()
	labelKey := metav1.LabelSelectorQueryParam(corev1GV.String())
	obj, err := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods?" + labelKey + "=a%3Db":     runtime.EncodeOrDie(corev1Codec, pods),
		"/namespaces/test/services?" + labelKey + "=a%3Db": runtime.EncodeOrDie(corev1Codec, svc),
	})).
		LabelSelectorParam("a=b").
		NamespaceParam("test").
		ResourceTypeOrNameArgs(true, "pods,services").
		Flatten().
		Do().Object()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	list, ok := obj.(*v1.List)
	if !ok {
		t.Fatalf("unexpected object: %#v", obj)
	}
	// resource version differs between type lists, so it's not possible to get a single version.
	if list.ResourceVersion != "" || len(list.Items) != 3 {
		t.Errorf("unexpected list: %#v", list)
	}
}

func TestWatch(t *testing.T) {
	_, svc := testData()
	w, err := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/services?fieldSelector=metadata.name%3Dredis-master&resourceVersion=12&watch=true": watchBody(watch.Event{
			Type:   watch.Added,
			Object: &svc.Items[0],
		}),
	})).
		NamespaceParam("test").DefaultNamespace().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{"../../artifacts/guestbook/redis-master-service.yaml"}}).Flatten().
		Do().Watch("12")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	defer w.Stop()
	ch := w.ResultChan()
	select {
	case obj := <-ch:
		if obj.Type != watch.Added {
			t.Fatalf("unexpected watch event %#v", obj)
		}
		service, ok := obj.Object.(*v1.Service)
		if !ok {
			t.Fatalf("unexpected object: %#v", obj)
		}
		if service.Name != "baz" || service.ResourceVersion != "12" {
			t.Errorf("unexpected service: %#v", service)
		}
	}
}

func TestWatchMultipleError(t *testing.T) {
	_, err := newDefaultBuilder().
		NamespaceParam("test").DefaultNamespace().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{"../../artifacts/guestbook/redis-master-controller.yaml"}}).Flatten().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{"../../artifacts/guestbook/redis-master-controller.yaml"}}).Flatten().
		Do().Watch("")

	if err == nil {
		t.Fatalf("unexpected non-error")
	}
}

func TestLatest(t *testing.T) {
	r, _, _ := streamTestData()
	newPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "13"},
	}
	newPod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "14"},
	}
	newSvc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "15"},
	}

	b := newDefaultBuilderWith(fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo":     runtime.EncodeOrDie(corev1Codec, newPod),
		"/namespaces/test/pods/bar":     runtime.EncodeOrDie(corev1Codec, newPod2),
		"/namespaces/test/services/baz": runtime.EncodeOrDie(corev1Codec, newSvc),
	})).
		NamespaceParam("other").Stream(r, "STDIN").Flatten().Latest()

	test := &testVisitor{}
	singleItemImplied := false

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err != nil || singleItemImplied || len(test.Infos) != 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}
	if !apiequality.Semantic.DeepDerivative([]runtime.Object{newPod, newPod2, newSvc}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}
}

func TestReceiveMultipleErrors(t *testing.T) {
	pods, svc := testData()

	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write([]byte(`{}`))
		w.Write([]byte(runtime.EncodeOrDie(corev1Codec, &pods.Items[0])))
	}()

	r2, w2 := io.Pipe()
	go func() {
		defer w2.Close()
		w2.Write([]byte(`{}`))
		w2.Write([]byte(runtime.EncodeOrDie(corev1Codec, &svc.Items[0])))
	}()

	b := newDefaultBuilder().
		Stream(r, "1").Stream(r2, "2").
		ContinueOnError()

	test := &testVisitor{}
	singleItemImplied := false

	err := b.Do().IntoSingleItemImplied(&singleItemImplied).Visit(test.Handle)
	if err == nil || singleItemImplied || len(test.Infos) != 2 {
		t.Fatalf("unexpected response: %v %t %#v", err, singleItemImplied, test.Infos)
	}

	errs, ok := err.(utilerrors.Aggregate)
	if !ok {
		t.Fatalf("unexpected error: %v", reflect.TypeOf(err))
	}
	if len(errs.Errors()) != 2 {
		t.Errorf("unexpected errors %v", errs)
	}
}

func TestHasNames(t *testing.T) {
	basename := filepath.Base(os.Args[0])
	tests := []struct {
		name            string
		args            []string
		expectedHasName bool
		expectedError   error
	}{
		{
			name:            "test1",
			args:            []string{""},
			expectedHasName: false,
			expectedError:   nil,
		},
		{
			name:            "test2",
			args:            []string{"rc"},
			expectedHasName: false,
			expectedError:   nil,
		},
		{
			name:            "test3",
			args:            []string{"rc,pod,svc"},
			expectedHasName: false,
			expectedError:   nil,
		},
		{
			name:            "test4",
			args:            []string{"rc/foo"},
			expectedHasName: true,
			expectedError:   nil,
		},
		{
			name:            "test5",
			args:            []string{"rc", "foo"},
			expectedHasName: true,
			expectedError:   nil,
		},
		{
			name:            "test6",
			args:            []string{"rc,pod,svc", "foo"},
			expectedHasName: true,
			expectedError:   nil,
		},
		{
			name:            "test7",
			args:            []string{"rc/foo", "rc/bar", "rc/zee"},
			expectedHasName: true,
			expectedError:   nil,
		},
		{
			name:            "test8",
			args:            []string{"rc/foo", "bar"},
			expectedHasName: false,
			expectedError:   fmt.Errorf("there is no need to specify a resource type as a separate argument when passing arguments in resource/name form (e.g. '" + basename + " get resource/<resource_name>' instead of '" + basename + " get resource resource/<resource_name>'"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hasNames, err := HasNames(tt.args)
			if !reflect.DeepEqual(tt.expectedError, err) {
				t.Errorf("expected HasName to error:\n%s\tgot:\n%s", tt.expectedError, err)
			}
			if hasNames != tt.expectedHasName {
				t.Errorf("expected HasName to return %v for %s", tt.expectedHasName, tt.args)
			}
		})
	}
}
