/*
Copyright 2016 The Kubernetes Authors.

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

package set

import (
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
)

func TestUpdateSelectorForObjectTypes(t *testing.T) {
	before := metav1.LabelSelector{MatchLabels: map[string]string{"fee": "true"},
		MatchExpressions: []metav1.LabelSelectorRequirement{
			{
				Key:      "foo",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"on", "yes"},
			},
		}}

	rc := v1.ReplicationController{}
	ser := v1.Service{}
	dep := extensionsv1beta1.Deployment{Spec: extensionsv1beta1.DeploymentSpec{Selector: &before}}
	ds := extensionsv1beta1.DaemonSet{Spec: extensionsv1beta1.DaemonSetSpec{Selector: &before}}
	rs := extensionsv1beta1.ReplicaSet{Spec: extensionsv1beta1.ReplicaSetSpec{Selector: &before}}
	job := batchv1.Job{Spec: batchv1.JobSpec{Selector: &before}}
	pvc := v1.PersistentVolumeClaim{Spec: v1.PersistentVolumeClaimSpec{Selector: &before}}
	sa := v1.ServiceAccount{}
	type args struct {
		obj      runtime.Object
		selector metav1.LabelSelector
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{name: "rc",
			args: args{
				obj:      &rc,
				selector: metav1.LabelSelector{},
			},
			wantErr: true,
		},
		{name: "ser",
			args: args{
				obj:      &ser,
				selector: metav1.LabelSelector{},
			},
			wantErr: false,
		},
		{name: "dep",
			args: args{
				obj:      &dep,
				selector: metav1.LabelSelector{},
			},
			wantErr: true,
		},
		{name: "ds",
			args: args{
				obj:      &ds,
				selector: metav1.LabelSelector{},
			},
			wantErr: true,
		},
		{name: "rs",
			args: args{
				obj:      &rs,
				selector: metav1.LabelSelector{},
			},
			wantErr: true,
		},
		{name: "job",
			args: args{
				obj:      &job,
				selector: metav1.LabelSelector{},
			},
			wantErr: true,
		},
		{name: "pvc - no updates",
			args: args{
				obj:      &pvc,
				selector: metav1.LabelSelector{},
			},
			wantErr: true,
		},
		{name: "sa - no selector",
			args: args{
				obj:      &sa,
				selector: metav1.LabelSelector{},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		if err := updateSelectorForObject(tt.args.obj, tt.args.selector); (err != nil) != tt.wantErr {
			t.Errorf("%q. updateSelectorForObject() error = %v, wantErr %v", tt.name, err, tt.wantErr)
		}
	}
}

func TestUpdateNewSelectorValuesForObject(t *testing.T) {
	ser := v1.Service{}
	type args struct {
		obj      runtime.Object
		selector metav1.LabelSelector
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{name: "empty",
			args: args{
				obj: &ser,
				selector: metav1.LabelSelector{
					MatchLabels:      map[string]string{},
					MatchExpressions: []metav1.LabelSelectorRequirement{},
				},
			},
			wantErr: false,
		},
		{name: "label-only",
			args: args{
				obj: &ser,
				selector: metav1.LabelSelector{
					MatchLabels:      map[string]string{"b": "u"},
					MatchExpressions: []metav1.LabelSelectorRequirement{},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		if err := updateSelectorForObject(tt.args.obj, tt.args.selector); (err != nil) != tt.wantErr {
			t.Errorf("%q. updateSelectorForObject() error = %v, wantErr %v", tt.name, err, tt.wantErr)
		}

		assert.EqualValues(t, tt.args.selector.MatchLabels, ser.Spec.Selector, tt.name)

	}
}

func TestUpdateOldSelectorValuesForObject(t *testing.T) {
	ser := v1.Service{Spec: v1.ServiceSpec{Selector: map[string]string{"fee": "true"}}}
	type args struct {
		obj      runtime.Object
		selector metav1.LabelSelector
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{name: "empty",
			args: args{
				obj: &ser,
				selector: metav1.LabelSelector{
					MatchLabels:      map[string]string{},
					MatchExpressions: []metav1.LabelSelectorRequirement{},
				},
			},
			wantErr: false,
		},
		{name: "label-only",
			args: args{
				obj: &ser,
				selector: metav1.LabelSelector{
					MatchLabels:      map[string]string{"fee": "false", "x": "y"},
					MatchExpressions: []metav1.LabelSelectorRequirement{},
				},
			},
			wantErr: false,
		},
		{name: "expr-only - err",
			args: args{
				obj: &ser,
				selector: metav1.LabelSelector{
					MatchLabels: map[string]string{},
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "a",
							Operator: "In",
							Values:   []string{"x", "y"},
						},
					},
				},
			},
			wantErr: true,
		},
		{name: "both - err",
			args: args{
				obj: &ser,
				selector: metav1.LabelSelector{
					MatchLabels: map[string]string{"b": "u"},
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "a",
							Operator: "In",
							Values:   []string{"x", "y"},
						},
					},
				},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		err := updateSelectorForObject(tt.args.obj, tt.args.selector)
		if (err != nil) != tt.wantErr {
			t.Errorf("%q. updateSelectorForObject() error = %v, wantErr %v", tt.name, err, tt.wantErr)
		} else if !tt.wantErr {
			assert.EqualValues(t, tt.args.selector.MatchLabels, ser.Spec.Selector, tt.name)
		}
	}
}

func TestGetResourcesAndSelector(t *testing.T) {
	type args struct {
		args []string
	}
	tests := []struct {
		name          string
		args          args
		wantResources []string
		wantSelector  *metav1.LabelSelector
		wantErr       bool
	}{
		{
			name:          "basic match",
			args:          args{args: []string{"rc/foo", "healthy=true"}},
			wantResources: []string{"rc/foo"},
			wantErr:       false,
			wantSelector: &metav1.LabelSelector{
				MatchLabels:      map[string]string{"healthy": "true"},
				MatchExpressions: []metav1.LabelSelectorRequirement{},
			},
		},
		{
			name:          "basic expression",
			args:          args{args: []string{"rc/foo", "buildType notin (debug, test)"}},
			wantResources: []string{"rc/foo"},
			wantErr:       false,
			wantSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{
						Key:      "buildType",
						Operator: "NotIn",
						Values:   []string{"debug", "test"},
					},
				},
			},
		},
		{
			name:          "selector error",
			args:          args{args: []string{"rc/foo", "buildType notthis (debug, test)"}},
			wantResources: []string{"rc/foo"},
			wantErr:       true,
			wantSelector: &metav1.LabelSelector{
				MatchLabels:      map[string]string{},
				MatchExpressions: []metav1.LabelSelectorRequirement{},
			},
		},
		{
			name:          "no resource and selector",
			args:          args{args: []string{}},
			wantResources: []string{},
			wantErr:       false,
			wantSelector:  nil,
		},
	}
	for _, tt := range tests {
		gotResources, gotSelector, err := getResourcesAndSelector(tt.args.args)
		if err != nil {
			if !tt.wantErr {
				t.Errorf("%q. getResourcesAndSelector() error = %v, wantErr %v", tt.name, err, tt.wantErr)
			}
			continue
		}
		if !reflect.DeepEqual(gotResources, tt.wantResources) {
			t.Errorf("%q. getResourcesAndSelector() gotResources = %v, want %v", tt.name, gotResources, tt.wantResources)
		}
		if !reflect.DeepEqual(gotSelector, tt.wantSelector) {
			t.Errorf("%q. getResourcesAndSelector() gotSelector = %v, want %v", tt.name, gotSelector, tt.wantSelector)
		}
	}
}

func TestSelectorTest(t *testing.T) {
	info := &resource.Info{
		Object: &v1.Service{
			TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Service"},
			ObjectMeta: metav1.ObjectMeta{Namespace: "some-ns", Name: "cassandra"},
		},
	}

	labelToSet, err := metav1.ParseToLabelSelector("environment=qa")
	if err != nil {
		t.Fatal(err)
	}

	iostreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	o := &SetSelectorOptions{
		selector:       labelToSet,
		ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(info),
		Recorder:       genericclioptions.NoopRecorder{},
		PrintObj:       (&printers.NamePrinter{}).PrintObj,
		IOStreams:      iostreams,
	}

	if err := o.RunSelector(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "service/cassandra") {
		t.Errorf("did not set selector: %s", buf.String())
	}
}
