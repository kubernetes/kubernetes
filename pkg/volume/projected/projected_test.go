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

package projected

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	clitesting "k8s.io/client-go/testing"
	pkgauthenticationv1 "k8s.io/kubernetes/pkg/apis/authentication/v1"
	pkgcorev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/emptydir"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
)

func TestCollectDataWithSecret(t *testing.T) {
	caseMappingMode := int32(0400)
	cases := []struct {
		name     string
		mappings []v1.KeyToPath
		secret   *v1.Secret
		mode     int32
		optional bool
		payload  map[string]util.FileProjection
		success  bool
	}{
		{
			name: "no overrides",
			secret: &v1.Secret{
				Data: map[string][]byte{
					"foo": []byte("foo"),
					"bar": []byte("bar"),
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"foo": {Data: []byte("foo"), Mode: 0644},
				"bar": {Data: []byte("bar"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "basic 1",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/foo.txt",
				},
			},
			secret: &v1.Secret{
				Data: map[string][]byte{
					"foo": []byte("foo"),
					"bar": []byte("bar"),
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"path/to/foo.txt": {Data: []byte("foo"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "subdirs",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/1/2/3/foo.txt",
				},
			},
			secret: &v1.Secret{
				Data: map[string][]byte{
					"foo": []byte("foo"),
					"bar": []byte("bar"),
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"path/to/1/2/3/foo.txt": {Data: []byte("foo"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "subdirs 2",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/1/2/3/foo.txt",
				},
			},
			secret: &v1.Secret{
				Data: map[string][]byte{
					"foo": []byte("foo"),
					"bar": []byte("bar"),
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"path/to/1/2/3/foo.txt": {Data: []byte("foo"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "subdirs 3",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/1/2/3/foo.txt",
				},
				{
					Key:  "bar",
					Path: "another/path/to/the/esteemed/bar.bin",
				},
			},
			secret: &v1.Secret{
				Data: map[string][]byte{
					"foo": []byte("foo"),
					"bar": []byte("bar"),
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"path/to/1/2/3/foo.txt":                {Data: []byte("foo"), Mode: 0644},
				"another/path/to/the/esteemed/bar.bin": {Data: []byte("bar"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "non existent key",
			mappings: []v1.KeyToPath{
				{
					Key:  "zab",
					Path: "path/to/foo.txt",
				},
			},
			secret: &v1.Secret{
				Data: map[string][]byte{
					"foo": []byte("foo"),
					"bar": []byte("bar"),
				},
			},
			mode:    0644,
			success: false,
		},
		{
			name: "mapping with Mode",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "foo.txt",
					Mode: &caseMappingMode,
				},
				{
					Key:  "bar",
					Path: "bar.bin",
					Mode: &caseMappingMode,
				},
			},
			secret: &v1.Secret{
				Data: map[string][]byte{
					"foo": []byte("foo"),
					"bar": []byte("bar"),
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"foo.txt": {Data: []byte("foo"), Mode: caseMappingMode},
				"bar.bin": {Data: []byte("bar"), Mode: caseMappingMode},
			},
			success: true,
		},
		{
			name: "mapping with defaultMode",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "foo.txt",
				},
				{
					Key:  "bar",
					Path: "bar.bin",
				},
			},
			secret: &v1.Secret{
				Data: map[string][]byte{
					"foo": []byte("foo"),
					"bar": []byte("bar"),
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"foo.txt": {Data: []byte("foo"), Mode: 0644},
				"bar.bin": {Data: []byte("bar"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "optional non existent key",
			mappings: []v1.KeyToPath{
				{
					Key:  "zab",
					Path: "path/to/foo.txt",
				},
			},
			secret: &v1.Secret{
				Data: map[string][]byte{
					"foo": []byte("foo"),
					"bar": []byte("bar"),
				},
			},
			mode:     0644,
			optional: true,
			payload:  map[string]util.FileProjection{},
			success:  true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {

			testNamespace := "test_projected_namespace"
			tc.secret.ObjectMeta = metav1.ObjectMeta{
				Namespace: testNamespace,
				Name:      tc.name,
			}

			source := makeProjection(tc.name, tc.mode, "secret")
			source.Sources[0].Secret.Items = tc.mappings
			source.Sources[0].Secret.Optional = &tc.optional

			testPodUID := types.UID("test_pod_uid")
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, UID: testPodUID}}
			client := fake.NewSimpleClientset(tc.secret)
			tempDir, host := newTestHost(t, client)
			defer os.RemoveAll(tempDir)
			var myVolumeMounter = projectedVolumeMounter{
				projectedVolume: &projectedVolume{
					sources: source.Sources,
					podUID:  pod.UID,
					plugin: &projectedPlugin{
						host:      host,
						getSecret: host.GetSecretFunc(),
					},
				},
				source: *source,
				pod:    pod,
			}

			actualPayload, err := myVolumeMounter.collectData()
			if err != nil && tc.success {
				t.Errorf("%v: unexpected failure making payload: %v", tc.name, err)
				return
			}
			if err == nil && !tc.success {
				t.Errorf("%v: unexpected success making payload", tc.name)
				return
			}
			if !tc.success {
				return
			}
			if e, a := tc.payload, actualPayload; !reflect.DeepEqual(e, a) {
				t.Errorf("%v: expected and actual payload do not match", tc.name)
			}
		})
	}
}

func TestCollectDataWithConfigMap(t *testing.T) {
	caseMappingMode := int32(0400)
	cases := []struct {
		name      string
		mappings  []v1.KeyToPath
		configMap *v1.ConfigMap
		mode      int32
		optional  bool
		payload   map[string]util.FileProjection
		success   bool
	}{
		{
			name: "no overrides",
			configMap: &v1.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"foo": {Data: []byte("foo"), Mode: 0644},
				"bar": {Data: []byte("bar"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "basic 1",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/foo.txt",
				},
			},
			configMap: &v1.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"path/to/foo.txt": {Data: []byte("foo"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "subdirs",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/1/2/3/foo.txt",
				},
			},
			configMap: &v1.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"path/to/1/2/3/foo.txt": {Data: []byte("foo"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "subdirs 2",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/1/2/3/foo.txt",
				},
			},
			configMap: &v1.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"path/to/1/2/3/foo.txt": {Data: []byte("foo"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "subdirs 3",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/1/2/3/foo.txt",
				},
				{
					Key:  "bar",
					Path: "another/path/to/the/esteemed/bar.bin",
				},
			},
			configMap: &v1.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"path/to/1/2/3/foo.txt":                {Data: []byte("foo"), Mode: 0644},
				"another/path/to/the/esteemed/bar.bin": {Data: []byte("bar"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "non existent key",
			mappings: []v1.KeyToPath{
				{
					Key:  "zab",
					Path: "path/to/foo.txt",
				},
			},
			configMap: &v1.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			mode:    0644,
			success: false,
		},
		{
			name: "mapping with Mode",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "foo.txt",
					Mode: &caseMappingMode,
				},
				{
					Key:  "bar",
					Path: "bar.bin",
					Mode: &caseMappingMode,
				},
			},
			configMap: &v1.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"foo.txt": {Data: []byte("foo"), Mode: caseMappingMode},
				"bar.bin": {Data: []byte("bar"), Mode: caseMappingMode},
			},
			success: true,
		},
		{
			name: "mapping with defaultMode",
			mappings: []v1.KeyToPath{
				{
					Key:  "foo",
					Path: "foo.txt",
				},
				{
					Key:  "bar",
					Path: "bar.bin",
				},
			},
			configMap: &v1.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"foo.txt": {Data: []byte("foo"), Mode: 0644},
				"bar.bin": {Data: []byte("bar"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "optional non existent key",
			mappings: []v1.KeyToPath{
				{
					Key:  "zab",
					Path: "path/to/foo.txt",
				},
			},
			configMap: &v1.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			mode:     0644,
			optional: true,
			payload:  map[string]util.FileProjection{},
			success:  true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testNamespace := "test_projected_namespace"
			tc.configMap.ObjectMeta = metav1.ObjectMeta{
				Namespace: testNamespace,
				Name:      tc.name,
			}

			source := makeProjection(tc.name, tc.mode, "configMap")
			source.Sources[0].ConfigMap.Items = tc.mappings
			source.Sources[0].ConfigMap.Optional = &tc.optional

			testPodUID := types.UID("test_pod_uid")
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, UID: testPodUID}}
			client := fake.NewSimpleClientset(tc.configMap)
			tempDir, host := newTestHost(t, client)
			defer os.RemoveAll(tempDir)
			var myVolumeMounter = projectedVolumeMounter{
				projectedVolume: &projectedVolume{
					sources: source.Sources,
					podUID:  pod.UID,
					plugin: &projectedPlugin{
						host:         host,
						getConfigMap: host.GetConfigMapFunc(),
					},
				},
				source: *source,
				pod:    pod,
			}

			actualPayload, err := myVolumeMounter.collectData()
			if err != nil && tc.success {
				t.Errorf("%v: unexpected failure making payload: %v", tc.name, err)
				return
			}
			if err == nil && !tc.success {
				t.Errorf("%v: unexpected success making payload", tc.name)
				return
			}
			if !tc.success {
				return
			}
			if e, a := tc.payload, actualPayload; !reflect.DeepEqual(e, a) {
				t.Errorf("%v: expected and actual payload do not match", tc.name)
			}
		})
	}
}

func TestCollectDataWithDownwardAPI(t *testing.T) {
	testNamespace := "test_projected_namespace"
	testPodUID := types.UID("test_pod_uid")
	testPodName := "podName"

	cases := []struct {
		name       string
		volumeFile []v1.DownwardAPIVolumeFile
		pod        *v1.Pod
		mode       int32
		payload    map[string]util.FileProjection
		success    bool
	}{
		{
			name: "annotation",
			volumeFile: []v1.DownwardAPIVolumeFile{
				{Path: "annotation", FieldRef: &v1.ObjectFieldSelector{
					FieldPath: "metadata.annotations['a1']"}}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPodName,
					Namespace: testNamespace,
					Annotations: map[string]string{
						"a1": "value1",
						"a2": "value2",
					},
					UID: testPodUID},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"annotation": {Data: []byte("value1"), Mode: 0644},
			},
			success: true,
		},
		{
			name: "annotation-error",
			volumeFile: []v1.DownwardAPIVolumeFile{
				{Path: "annotation", FieldRef: &v1.ObjectFieldSelector{
					FieldPath: "metadata.annotations['']"}}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPodName,
					Namespace: testNamespace,
					Annotations: map[string]string{
						"a1": "value1",
						"a2": "value2",
					},
					UID: testPodUID},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"annotation": {Data: []byte("does-not-matter-because-this-test-case-will-fail-anyway"), Mode: 0644},
			},
			success: false,
		},
		{
			name: "labels",
			volumeFile: []v1.DownwardAPIVolumeFile{
				{Path: "labels", FieldRef: &v1.ObjectFieldSelector{
					FieldPath: "metadata.labels"}}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPodName,
					Namespace: testNamespace,
					Labels: map[string]string{
						"key1": "value1",
						"key2": "value2"},
					UID: testPodUID},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"labels": {Data: []byte("key1=\"value1\"\nkey2=\"value2\""), Mode: 0644},
			},
			success: true,
		},
		{
			name: "annotations",
			volumeFile: []v1.DownwardAPIVolumeFile{
				{Path: "annotations", FieldRef: &v1.ObjectFieldSelector{
					FieldPath: "metadata.annotations"}}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPodName,
					Namespace: testNamespace,
					Annotations: map[string]string{
						"a1": "value1",
						"a2": "value2"},
					UID: testPodUID},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"annotations": {Data: []byte("a1=\"value1\"\na2=\"value2\""), Mode: 0644},
			},
			success: true,
		},
		{
			name: "name",
			volumeFile: []v1.DownwardAPIVolumeFile{
				{Path: "name_file_name", FieldRef: &v1.ObjectFieldSelector{
					FieldPath: "metadata.name"}}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPodName,
					Namespace: testNamespace,
					UID:       testPodUID},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"name_file_name": {Data: []byte(testPodName), Mode: 0644},
			},
			success: true,
		},
		{
			name: "namespace",
			volumeFile: []v1.DownwardAPIVolumeFile{
				{Path: "namespace_file_name", FieldRef: &v1.ObjectFieldSelector{
					FieldPath: "metadata.namespace"}}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPodName,
					Namespace: testNamespace,
					UID:       testPodUID},
			},
			mode: 0644,
			payload: map[string]util.FileProjection{
				"namespace_file_name": {Data: []byte(testNamespace), Mode: 0644},
			},
			success: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			source := makeProjection("", tc.mode, "downwardAPI")
			source.Sources[0].DownwardAPI.Items = tc.volumeFile

			client := fake.NewSimpleClientset(tc.pod)
			tempDir, host := newTestHost(t, client)
			defer os.RemoveAll(tempDir)
			var myVolumeMounter = projectedVolumeMounter{
				projectedVolume: &projectedVolume{
					sources: source.Sources,
					podUID:  tc.pod.UID,
					plugin: &projectedPlugin{
						host: host,
					},
				},
				source: *source,
				pod:    tc.pod,
			}

			actualPayload, err := myVolumeMounter.collectData()
			if err != nil && tc.success {
				t.Errorf("%v: unexpected failure making payload: %v", tc.name, err)
				return
			}
			if err == nil && !tc.success {
				t.Errorf("%v: unexpected success making payload", tc.name)
				return
			}
			if !tc.success {
				return
			}
			if e, a := tc.payload, actualPayload; !reflect.DeepEqual(e, a) {
				t.Errorf("%v: expected and actual payload do not match", tc.name)
			}
		})

	}
}

func TestCollectDataWithServiceAccountToken(t *testing.T) {
	scheme := runtime.NewScheme()
	utilruntime.Must(pkgauthenticationv1.RegisterDefaults(scheme))
	utilruntime.Must(pkgcorev1.RegisterDefaults(scheme))

	minute := int64(60)
	cases := []struct {
		name       string
		svcacct    string
		audience   string
		expiration *int64
		path       string

		payload map[string]util.FileProjection
	}{
		{
			name:       "test good service account",
			audience:   "https://example.com",
			path:       "token",
			expiration: &minute,

			payload: map[string]util.FileProjection{
				"token": {Data: []byte("test_projected_namespace:foo:60:[https://example.com]"), Mode: 0600},
			},
		},
		{
			name:       "test good service account other path",
			audience:   "https://example.com",
			path:       "other-token",
			expiration: &minute,

			payload: map[string]util.FileProjection{
				"other-token": {Data: []byte("test_projected_namespace:foo:60:[https://example.com]"), Mode: 0600},
			},
		},
		{
			name:       "test good service account defaults audience",
			path:       "token",
			expiration: &minute,

			payload: map[string]util.FileProjection{
				"token": {Data: []byte("test_projected_namespace:foo:60:[https://api]"), Mode: 0600},
			},
		},
		{
			name:     "test good service account defaults expiration",
			audience: "https://example.com",
			path:     "token",

			payload: map[string]util.FileProjection{
				"token": {Data: []byte("test_projected_namespace:foo:3600:[https://example.com]"), Mode: 0600},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testNamespace := "test_projected_namespace"
			source := makeProjection(tc.name, 0600, "serviceAccountToken")
			source.Sources[0].ServiceAccountToken.Audience = tc.audience
			source.Sources[0].ServiceAccountToken.ExpirationSeconds = tc.expiration
			source.Sources[0].ServiceAccountToken.Path = tc.path

			testPodUID := types.UID("test_pod_uid")
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, UID: testPodUID},
				Spec:       v1.PodSpec{ServiceAccountName: "foo"},
			}
			scheme.Default(pod)

			client := &fake.Clientset{}
			client.AddReactor("create", "serviceaccounts", clitesting.ReactionFunc(func(action clitesting.Action) (bool, runtime.Object, error) {
				tr := action.(clitesting.CreateAction).GetObject().(*authenticationv1.TokenRequest)
				scheme.Default(tr)
				if len(tr.Spec.Audiences) == 0 {
					tr.Spec.Audiences = []string{"https://api"}
				}
				tr.Status.Token = fmt.Sprintf("%v:%v:%d:%v", action.GetNamespace(), "foo", *tr.Spec.ExpirationSeconds, tr.Spec.Audiences)
				return true, tr, nil
			}))

			tempDir, host := newTestHost(t, client)
			defer os.RemoveAll(tempDir)

			var myVolumeMounter = projectedVolumeMounter{
				projectedVolume: &projectedVolume{
					sources: source.Sources,
					podUID:  pod.UID,
					plugin: &projectedPlugin{
						host:                   host,
						getServiceAccountToken: host.GetServiceAccountTokenFunc(),
					},
				},
				source: *source,
				pod:    pod,
			}

			actualPayload, err := myVolumeMounter.collectData()
			if err != nil {
				t.Fatalf("unexpected failure making payload: %v", err)
			}
			if e, a := tc.payload, actualPayload; !reflect.DeepEqual(e, a) {
				t.Errorf("expected and actual payload do not match:\n%s", diff.ObjectReflectDiff(e, a))
			}
		})
	}
}

func newTestHost(t *testing.T, clientset clientset.Interface) (string, volume.VolumeHost) {
	tempDir, err := ioutil.TempDir("/tmp", "projected_volume_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}

	return tempDir, volumetest.NewFakeVolumeHost(tempDir, clientset, emptydir.ProbeVolumePlugins())
}

func TestCanSupport(t *testing.T) {
	pluginMgr := volume.VolumePluginMgr{}
	tempDir, host := newTestHost(t, nil)
	defer os.RemoveAll(tempDir)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plugin, err := pluginMgr.FindPluginByName(projectedPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plugin.GetPluginName() != projectedPluginName {
		t.Errorf("Wrong name: %s", plugin.GetPluginName())
	}
	if !plugin.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{Projected: &v1.ProjectedVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if plugin.CanSupport(&volume.Spec{}) {
		t.Errorf("Expected false")
	}
}

func TestPlugin(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_projected_namespace"
		testName       = "test_projected_name"

		volumeSpec    = makeVolumeSpec(testVolumeName, testName, 0644)
		secret        = makeSecret(testNamespace, testName)
		client        = fake.NewSimpleClientset(&secret)
		pluginMgr     = volume.VolumePluginMgr{}
		rootDir, host = newTestHost(t, client)
	)
	defer os.RemoveAll(rootDir)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plugin, err := pluginMgr.FindPluginByName(projectedPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, UID: testPodUID}}
	mounter, err := plugin.NewMounter(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volumePath := mounter.GetPath()
	if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid/volumes/kubernetes.io~projected/%s", testVolumeName)) {
		t.Errorf("Got unexpected path: %s", volumePath)
	}

	err = mounter.SetUp(volume.MounterArgs{})
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	// secret volume should create its own empty wrapper path
	podWrapperMetadataDir := fmt.Sprintf("%v/pods/test_pod_uid/plugins/kubernetes.io~empty-dir/wrapped_test_volume_name", rootDir)

	if _, err := os.Stat(podWrapperMetadataDir); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, empty-dir wrapper path is not created: %s", podWrapperMetadataDir)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	doTestSecretDataInVolume(volumePath, secret, t)
	defer doTestCleanAndTeardown(plugin, testPodUID, testVolumeName, volumePath, t)
}

func TestInvalidPathProjected(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_projected_namespace"
		testName       = "test_projected_name"

		volumeSpec    = makeVolumeSpec(testVolumeName, testName, 0644)
		secret        = makeSecret(testNamespace, testName)
		client        = fake.NewSimpleClientset(&secret)
		pluginMgr     = volume.VolumePluginMgr{}
		rootDir, host = newTestHost(t, client)
	)
	volumeSpec.Projected.Sources[0].Secret.Items = []v1.KeyToPath{
		{Key: "missing", Path: "missing"},
	}

	defer os.RemoveAll(rootDir)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plugin, err := pluginMgr.FindPluginByName(projectedPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, UID: testPodUID}}
	mounter, err := plugin.NewMounter(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volumePath := mounter.GetPath()
	if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid/volumes/kubernetes.io~projected/%s", testVolumeName)) {
		t.Errorf("Got unexpected path: %s", volumePath)
	}

	var mounterArgs volume.MounterArgs
	err = mounter.SetUp(mounterArgs)
	if err == nil {
		t.Errorf("Expected error while setting up secret")
	}

	_, err = os.Stat(volumePath)
	if err == nil {
		t.Errorf("Expected path %s to not exist", volumePath)
	}
}

// Test the case where the plugin's ready file exists, but the volume dir is not a
// mountpoint, which is the state the system will be in after reboot.  The dir
// should be mounter and the secret data written to it.
func TestPluginReboot(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid3")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_secret_namespace"
		testName       = "test_secret_name"

		volumeSpec    = makeVolumeSpec(testVolumeName, testName, 0644)
		secret        = makeSecret(testNamespace, testName)
		client        = fake.NewSimpleClientset(&secret)
		pluginMgr     = volume.VolumePluginMgr{}
		rootDir, host = newTestHost(t, client)
	)
	defer os.RemoveAll(rootDir)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plugin, err := pluginMgr.FindPluginByName(projectedPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, UID: testPodUID}}
	mounter, err := plugin.NewMounter(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	podMetadataDir := fmt.Sprintf("%v/pods/test_pod_uid3/plugins/kubernetes.io~projected/test_volume_name", rootDir)
	util.SetReady(podMetadataDir)
	volumePath := mounter.GetPath()
	if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid3/volumes/kubernetes.io~projected/test_volume_name")) {
		t.Errorf("Got unexpected path: %s", volumePath)
	}

	err = mounter.SetUp(volume.MounterArgs{})
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	doTestSecretDataInVolume(volumePath, secret, t)
	doTestCleanAndTeardown(plugin, testPodUID, testVolumeName, volumePath, t)
}

func TestPluginOptional(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_secret_namespace"
		testName       = "test_secret_name"
		trueVal        = true

		volumeSpec    = makeVolumeSpec(testVolumeName, testName, 0644)
		client        = fake.NewSimpleClientset()
		pluginMgr     = volume.VolumePluginMgr{}
		rootDir, host = newTestHost(t, client)
	)
	volumeSpec.VolumeSource.Projected.Sources[0].Secret.Optional = &trueVal
	defer os.RemoveAll(rootDir)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plugin, err := pluginMgr.FindPluginByName(projectedPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, UID: testPodUID}}
	mounter, err := plugin.NewMounter(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volumePath := mounter.GetPath()
	if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid/volumes/kubernetes.io~projected/test_volume_name")) {
		t.Errorf("Got unexpected path: %s", volumePath)
	}

	err = mounter.SetUp(volume.MounterArgs{})
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	// secret volume should create its own empty wrapper path
	podWrapperMetadataDir := fmt.Sprintf("%v/pods/test_pod_uid/plugins/kubernetes.io~empty-dir/wrapped_test_volume_name", rootDir)

	if _, err := os.Stat(podWrapperMetadataDir); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, empty-dir wrapper path is not created: %s", podWrapperMetadataDir)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	datadirSymlink := filepath.Join(volumePath, "..data")
	datadir, err := os.Readlink(datadirSymlink)
	if err != nil && os.IsNotExist(err) {
		t.Fatalf("couldn't find volume path's data dir, %s", datadirSymlink)
	} else if err != nil {
		t.Fatalf("couldn't read symlink, %s", datadirSymlink)
	}
	datadirPath := filepath.Join(volumePath, datadir)

	infos, err := ioutil.ReadDir(volumePath)
	if err != nil {
		t.Fatalf("couldn't find volume path, %s", volumePath)
	}
	if len(infos) != 0 {
		for _, fi := range infos {
			if fi.Name() != "..data" && fi.Name() != datadir {
				t.Errorf("empty data volume directory, %s, is not empty. Contains: %s", datadirSymlink, fi.Name())
			}
		}
	}

	infos, err = ioutil.ReadDir(datadirPath)
	if err != nil {
		t.Fatalf("couldn't find volume data path, %s", datadirPath)
	}
	if len(infos) != 0 {
		t.Errorf("empty data directory, %s, is not empty. Contains: %s", datadirSymlink, infos[0].Name())
	}

	defer doTestCleanAndTeardown(plugin, testPodUID, testVolumeName, volumePath, t)
}

func TestPluginOptionalKeys(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_secret_namespace"
		testName       = "test_secret_name"
		trueVal        = true

		volumeSpec    = makeVolumeSpec(testVolumeName, testName, 0644)
		secret        = makeSecret(testNamespace, testName)
		client        = fake.NewSimpleClientset(&secret)
		pluginMgr     = volume.VolumePluginMgr{}
		rootDir, host = newTestHost(t, client)
	)
	volumeSpec.VolumeSource.Projected.Sources[0].Secret.Items = []v1.KeyToPath{
		{Key: "data-1", Path: "data-1"},
		{Key: "data-2", Path: "data-2"},
		{Key: "data-3", Path: "data-3"},
		{Key: "missing", Path: "missing"},
	}
	volumeSpec.VolumeSource.Projected.Sources[0].Secret.Optional = &trueVal
	defer os.RemoveAll(rootDir)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plugin, err := pluginMgr.FindPluginByName(projectedPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, UID: testPodUID}}
	mounter, err := plugin.NewMounter(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volumePath := mounter.GetPath()
	if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid/volumes/kubernetes.io~projected/test_volume_name")) {
		t.Errorf("Got unexpected path: %s", volumePath)
	}

	err = mounter.SetUp(volume.MounterArgs{})
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	// secret volume should create its own empty wrapper path
	podWrapperMetadataDir := fmt.Sprintf("%v/pods/test_pod_uid/plugins/kubernetes.io~empty-dir/wrapped_test_volume_name", rootDir)

	if _, err := os.Stat(podWrapperMetadataDir); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, empty-dir wrapper path is not created: %s", podWrapperMetadataDir)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	doTestSecretDataInVolume(volumePath, secret, t)
	defer doTestCleanAndTeardown(plugin, testPodUID, testVolumeName, volumePath, t)
}

func makeVolumeSpec(volumeName, name string, defaultMode int32) *v1.Volume {
	return &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			Projected: makeProjection(name, defaultMode, "secret"),
		},
	}
}

func makeSecret(namespace, name string) v1.Secret {
	return v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Data: map[string][]byte{
			"data-1": []byte("value-1"),
			"data-2": []byte("value-2"),
			"data-3": []byte("value-3"),
		},
	}
}

func makeProjection(name string, defaultMode int32, kind string) *v1.ProjectedVolumeSource {
	var item v1.VolumeProjection

	switch kind {
	case "configMap":
		item = v1.VolumeProjection{
			ConfigMap: &v1.ConfigMapProjection{
				LocalObjectReference: v1.LocalObjectReference{Name: name},
			},
		}
	case "secret":
		item = v1.VolumeProjection{
			Secret: &v1.SecretProjection{
				LocalObjectReference: v1.LocalObjectReference{Name: name},
			},
		}
	case "downwardAPI":
		item = v1.VolumeProjection{
			DownwardAPI: &v1.DownwardAPIProjection{},
		}
	case "serviceAccountToken":
		item = v1.VolumeProjection{
			ServiceAccountToken: &v1.ServiceAccountTokenProjection{},
		}
	}

	return &v1.ProjectedVolumeSource{
		Sources:     []v1.VolumeProjection{item},
		DefaultMode: &defaultMode,
	}
}

func doTestSecretDataInVolume(volumePath string, secret v1.Secret, t *testing.T) {
	for key, value := range secret.Data {
		secretDataHostPath := filepath.Join(volumePath, key)
		if _, err := os.Stat(secretDataHostPath); err != nil {
			t.Fatalf("SetUp() failed, couldn't find secret data on disk: %v", secretDataHostPath)
		} else {
			actualSecretBytes, err := ioutil.ReadFile(secretDataHostPath)
			if err != nil {
				t.Fatalf("Couldn't read secret data from: %v", secretDataHostPath)
			}

			actualSecretValue := string(actualSecretBytes)
			if string(value) != actualSecretValue {
				t.Errorf("Unexpected value; expected %q, got %q", value, actualSecretValue)
			}
		}
	}
}

func doTestCleanAndTeardown(plugin volume.VolumePlugin, podUID types.UID, testVolumeName, volumePath string, t *testing.T) {
	unmounter, err := plugin.NewUnmounter(testVolumeName, podUID)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volumePath)
	} else if !os.IsNotExist(err) {
		t.Errorf("TearDown() failed: %v", err)
	}
}
