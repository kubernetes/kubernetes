/*
Copyright 2017 The Kubernetes Authors.

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

package checkpoint

import (
	"fmt"
	"testing"

	"github.com/davecgh/go-spew/spew"

	apiv1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	kubeletconfigv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1alpha1"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
)

func TestNewConfigMapCheckpoint(t *testing.T) {
	cases := []struct {
		desc string
		cm   *apiv1.ConfigMap
		err  string
	}{
		{"nil v1/ConfigMap", nil, "must be non-nil"},
		{"empty v1/ConfigMap", &apiv1.ConfigMap{}, "must have a UID"},
		{"populated v1/ConfigMap",
			&apiv1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "name",
					UID:  types.UID("uid"),
				},
				Data: map[string]string{
					"key1": "value1",
					"key2": "value2",
				},
			}, ""},
	}

	for _, c := range cases {
		cpt, err := NewConfigMapCheckpoint(c.cm)
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		// underlying object should match the object passed in
		if !apiequality.Semantic.DeepEqual(cpt.object(), c.cm) {
			t.Errorf("case %q, expect Checkpoint %s but got %s", c.desc, spew.Sdump(c.cm), spew.Sdump(cpt))
		}
	}
}

func TestConfigMapCheckpointUID(t *testing.T) {
	_, kubeletCodecs, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []string{"", "uid", "376dfb73-56db-11e7-a01e-42010a800002"}
	for _, uidIn := range cases {
		cpt := &configMapCheckpoint{
			kubeletCodecs,
			&apiv1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{UID: types.UID(uidIn)},
			},
		}
		// UID method should return the correct value of the UID
		uidOut := cpt.UID()
		if uidIn != uidOut {
			t.Errorf("expect UID() to return %q, but got %q", uidIn, uidOut)
		}
	}
}

func TestConfigMapCheckpointParse(t *testing.T) {
	kubeletScheme, kubeletCodecs, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// get the built-in default configuration
	external := &kubeletconfigv1alpha1.KubeletConfiguration{}
	kubeletScheme.Default(external)
	defaultConfig := &kubeletconfig.KubeletConfiguration{}
	err = kubeletScheme.Convert(external, defaultConfig, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		desc   string
		cm     *apiv1.ConfigMap
		expect *kubeletconfig.KubeletConfiguration
		err    string
	}{
		{"empty data", &apiv1.ConfigMap{}, nil, "config was empty"},
		// missing kubelet key
		{"missing kubelet key", &apiv1.ConfigMap{Data: map[string]string{
			"bogus": "stuff"}}, nil, fmt.Sprintf("key %q not found", configMapConfigKey)},
		// invalid format
		{"invalid yaml", &apiv1.ConfigMap{Data: map[string]string{
			"kubelet": "*"}}, nil, "failed to decode"},
		{"invalid json", &apiv1.ConfigMap{Data: map[string]string{
			"kubelet": "{*"}}, nil, "failed to decode"},
		// invalid object
		{"missing kind", &apiv1.ConfigMap{Data: map[string]string{
			"kubelet": `{"apiVersion":"kubeletconfig/v1alpha1"}`}}, nil, "failed to decode"},
		{"missing version", &apiv1.ConfigMap{Data: map[string]string{
			"kubelet": `{"kind":"KubeletConfiguration"}`}}, nil, "failed to decode"},
		{"unregistered kind", &apiv1.ConfigMap{Data: map[string]string{
			"kubelet": `{"kind":"BogusKind","apiVersion":"kubeletconfig/v1alpha1"}`}}, nil, "failed to decode"},
		{"unregistered version", &apiv1.ConfigMap{Data: map[string]string{
			"kubelet": `{"kind":"KubeletConfiguration","apiVersion":"bogusversion"}`}}, nil, "failed to decode"},
		// empty object with correct kind and version should result in the defaults for that kind and version
		{"default from yaml", &apiv1.ConfigMap{Data: map[string]string{
			"kubelet": `kind: KubeletConfiguration
apiVersion: kubeletconfig/v1alpha1`}}, defaultConfig, ""},
		{"default from json", &apiv1.ConfigMap{Data: map[string]string{
			"kubelet": `{"kind":"KubeletConfiguration","apiVersion":"kubeletconfig/v1alpha1"}`}}, defaultConfig, ""},
	}
	for _, c := range cases {
		cpt := &configMapCheckpoint{kubeletCodecs, c.cm}
		kc, err := cpt.Parse()
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		// we expect the parsed configuration to match what we described in the ConfigMap
		if !apiequality.Semantic.DeepEqual(c.expect, kc) {
			t.Errorf("case %q, expect config %s but got %s", c.desc, spew.Sdump(c.expect), spew.Sdump(kc))
		}
	}
}

func TestConfigMapCheckpointEncode(t *testing.T) {
	_, kubeletCodecs, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// only one case, based on output from the existing encoder, and since
	// this is hard to test (key order isn't guaranteed), we should probably
	// just stick to this test case and mostly rely on the round-trip test.
	cases := []struct {
		desc   string
		cpt    *configMapCheckpoint
		expect string
	}{
		// we expect Checkpoints to be encoded as a json representation of the underlying API object
		{"one-key",
			&configMapCheckpoint{kubeletCodecs, &apiv1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "one-key"},
				Data:       map[string]string{"one": ""}}},
			`{"kind":"ConfigMap","apiVersion":"v1","metadata":{"name":"one-key","creationTimestamp":null},"data":{"one":""}}
`},
	}

	for _, c := range cases {
		data, err := c.cpt.Encode()
		// we don't expect any errors from encoding
		if utiltest.SkipRest(t, c.desc, err, "") {
			continue
		}
		if string(data) != c.expect {
			t.Errorf("case %q, expect encoding %q but got %q", c.desc, c.expect, string(data))
		}
	}
}

func TestConfigMapCheckpointRoundTrip(t *testing.T) {
	_, kubeletCodecs, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		desc      string
		cpt       *configMapCheckpoint
		decodeErr string
	}{
		// empty data
		{"empty data",
			&configMapCheckpoint{kubeletCodecs, &apiv1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "empty-data-sha256-e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
					UID:  "uid",
				},
				Data: map[string]string{}}},
			""},
		// two keys
		{"two keys",
			&configMapCheckpoint{kubeletCodecs, &apiv1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "two-keys-sha256-2bff03d6249c8a9dc9a1436d087c124741361ccfac6615b81b67afcff5c42431",
					UID:  "uid",
				},
				Data: map[string]string{"one": "", "two": "2"}}},
			""},
		// missing uid
		{"missing uid",
			&configMapCheckpoint{kubeletCodecs, &apiv1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "two-keys-sha256-2bff03d6249c8a9dc9a1436d087c124741361ccfac6615b81b67afcff5c42431",
					UID:  "",
				},
				Data: map[string]string{"one": "", "two": "2"}}},
			"must have a UID"},
	}
	for _, c := range cases {
		// we don't expect any errors from encoding
		data, err := c.cpt.Encode()
		if utiltest.SkipRest(t, c.desc, err, "") {
			continue
		}
		after, err := DecodeCheckpoint(data)
		if utiltest.SkipRest(t, c.desc, err, c.decodeErr) {
			continue
		}
		if !apiequality.Semantic.DeepEqual(c.cpt.object(), after.object()) {
			t.Errorf("case %q, expect round-trip result %s but got %s", c.desc, spew.Sdump(c.cpt), spew.Sdump(after))
		}
	}
}
