/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package template

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	kyaml "k8s.io/kubernetes/pkg/util/yaml"

	"github.com/stretchr/testify/assert"
)

type FakeRegistry struct {
	value       *extensions.Template
	err         error
	expectedCtx api.Context
	t           *testing.T
}

func (r *FakeRegistry) GetTemplate(ctx api.Context, templateName string) (*extensions.Template, error) {
	assert.Exactly(r.t, r.expectedCtx, ctx)
	if r.err != nil {
		return nil, r.err
	}
	return r.value, nil
}

func getObjects(t *testing.T, objs ...string) []runtime.RawExtension {
	values := []runtime.RawExtension{}
	for _, obj := range objs {
		json, err := kyaml.ToJSON([]byte(obj))
		assert.NoError(t, err)
		values = append(values, runtime.RawExtension{Raw: json})
	}
	return values
}

func getService(nameSuffix, port string) string {
	return fmt.Sprintf(`apiVersion: v1
kind: Service
metadata:
    name: service-name%s
spec:
    selector:
        app: type
    ports:
    - protocol: TCP
      port: %s`, nameSuffix, port)
}

func getDeployment(nameSuffix, replicas, stdin string) string {
	return fmt.Sprintf(`apiVersion: extensions/v1beta1
kind: Deployment
metadata:
    name: deployment-name%s
spec:
    replicas: %s
    template:
        kind: Pod
        apiVersion: v1
        metadata:
            name: pod-name%s
            labels:
                app: pod-label
        spec:
            containers:
            - image: nginx:1.9.7
              name: %s
              stdin: %s`, nameSuffix, replicas, nameSuffix, stdin, stdin)

}

func expectedDeployment(nameSuffix string, replicas int32, stdin bool) v1beta1.Deployment {
	return v1beta1.Deployment{
		TypeMeta: unversioned.TypeMeta{APIVersion: "extensions/v1beta1", Kind: "Deployment"},
		ObjectMeta: v1.ObjectMeta{Name: "deployment-name" + nameSuffix},
		Spec: v1beta1.DeploymentSpec{
			Replicas: &replicas,
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{Name: "pod-name" + nameSuffix, Labels: map[string]string{"app": "pod-label"}},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						v1.Container{
							Stdin: stdin,
							Image: "nginx:1.9.7",
							Name: "nginx",
						},
					},
				},
			},
		},
	}
}

func expectedService(nameSuffix string, port int32) v1.Service {
	return v1.Service{
		TypeMeta: unversioned.TypeMeta{APIVersion: "v1", Kind: "Service"},
		ObjectMeta: v1.ObjectMeta{Name: "service-name" + nameSuffix},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{
				v1.ServicePort{Protocol: "TCP", Port: port},
			},
			Selector: map[string]string{
				"app": "type",
			},
		},
	}
}

func NewTestProcessor(t *testing.T) (TemplateProcessor, *FakeRegistry, api.Context) {
	ctx := api.NewContext()
	r := &FakeRegistry{
		t:           t,
		expectedCtx: ctx,
	}
	scheme := runtime.NewScheme()
	v1.AddToScheme(scheme)
	v1beta1.AddToScheme(scheme)
	codecs := serializer.NewCodecFactory(scheme)
	return NewTemplateProcessor(r, codecs.UniversalDeserializer()), r, ctx
}

func TestProcessFailedLookup(t *testing.T) {
	p, r, ctx := NewTestProcessor(t)

	// Unable to lookup Template - Error
	r.err = fmt.Errorf("Could not find Template.")
	list, err := p.Process(ctx, &extensions.TemplateParameters{
		Name: "fake-name",
	})
	assert.Nil(t, list)
	assert.Error(t, err)
}

func TestProcessValidationNameNotSpecified(t *testing.T) {
	p, r, ctx := NewTestProcessor(t)

	r.value = &extensions.Template{
		Spec: extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				{
					Name: "PORT",
					Type: "integer",
				},
				{
					Name: "OTHER_PARAM",
				},
			},
			Objects: getObjects(t, getService("", "$(PORT)")),
		},
	}
	list, err := p.Process(ctx, &extensions.TemplateParameters{
		Name: "fake-name",
		ParameterValues: map[string]string{
			"PORT": "2",
			"OTHER_PARAM":   "hello",
		},
	})
	assert.NoError(t, err)
	assert.NotEmpty(t, list)

	// OTHER_PARAM note specified in Template - Error
	r.value = &extensions.Template{
		Spec: extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				extensions.Parameter{
					Name: "PORT",
					Type: "integer",
				},
			},
			Objects: getObjects(t, getService("", "$(PORT)")),
		},
	}
	list, err = p.Process(ctx, &extensions.TemplateParameters{
		Name: "fake-name",
		ParameterValues: map[string]string{
			"PORT": "2",
			"OTHER_PARAM":   "hello",
		},
	})
	assert.Nil(t, list)
	assert.Error(t, err)
}
//
func TestProcessValidationType(t *testing.T) {
	p, r, ctx := NewTestProcessor(t)

	// Base case - Success
	r.value = &extensions.Template{
		Spec: extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				extensions.Parameter{
					Name: "PORT",
					Type: "integer",
				},
			},
			Objects: getObjects(t, getService("", "$(PORT)")),
		},
	}
	list, err := p.Process(ctx, &extensions.TemplateParameters{
		Name: "fake-name",
		ParameterValues: map[string]string{
			"PORT": "2",
		},
	})
	assert.NoError(t, err)
	assert.NotEmpty(t, list)

	// Value is not an integer - Error
	r.value = &extensions.Template{
		Spec: extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				extensions.Parameter{
					Name: "PORT",
					Type: "integer",
				},
			},
			Objects: getObjects(t, getService("", "$(PORT)")),

		},
	}
	list, err = p.Process(ctx, &extensions.TemplateParameters{
		Name: "fake-name",
		ParameterValues: map[string]string{
			"PORT": "true",
		},
	})
	assert.Nil(t, list)
	assert.Error(t, err)

	// Value is not specified - Error
	r.value = &extensions.Template{
		Spec: extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				extensions.Parameter{
					Name: "PORT",
					Type: "integer",
				},
			},
			Objects: getObjects(t, getService("", "$(PORT)")),
		},
	}
	list, err = p.Process(ctx, &extensions.TemplateParameters{
		Name: "fake-name",
		ParameterValues: map[string]string{
			"PORT": "",
		},
	})
	assert.Nil(t, list)
	assert.Error(t, err)
}

 // Verify Required Params must have a non-empty value specified even if they are overriden
func TestProcessValidationRequired(t *testing.T) {
	p, r, ctx := NewTestProcessor(t)

	// Basecase - Ok
	r.value = &extensions.Template{
		Spec: extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				extensions.Parameter{
					Name:     "SUFFIX",
					Required: true,
					Value:    "hello",
				},
			},
			Objects: getObjects(t, getService("$(SUFFIX)", "80")),
		},
	}
	list, err := p.Process(ctx, &extensions.TemplateParameters{
		Name: "fake-name",
		ParameterValues: map[string]string{
			"SUFFIX": "world",
		},
	})
	assert.NoError(t, err)
	assert.NotEmpty(t, list)

	// Value not specified - Error
	r.value = &extensions.Template{
		Spec: extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				extensions.Parameter{
					Name:     "SUFFIX",
					Required: true,
				},
			},
			Objects: getObjects(t, getService("$(SUFFIX)", "80")),
		},
	}
	list, err = p.Process(ctx, &extensions.TemplateParameters{
		Name:            "fake-name",
		ParameterValues: map[string]string{},
	})
	assert.Nil(t, list)
	assert.Error(t, err)

	// Value overridden with empty string - Error
	r.value = &extensions.Template{
		Spec: extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				extensions.Parameter{
					Name:     "SUFFIX",
					Required: true,
					Value:    "hello",
				},
			},
			Objects: getObjects(t, getService("$(SUFFIX)", "$(PORT)")),
		},
	}
	list, err = p.Process(ctx, &extensions.TemplateParameters{
		Name: "fake-name",
		ParameterValues: map[string]string{
			"SUFFIX": "",
		},
	})
	assert.Nil(t, list)
	assert.Error(t, err)
}

func TestProcess(t *testing.T) {
	p, r, ctx := NewTestProcessor(t)

	fmt.Println(getDeployment("$(SUFFIX)", "$(REPLICAS)", "$(STDIN)"))
	r.value = &extensions.Template{
		Spec: extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				extensions.Parameter{
					Name: "PORT",
					Type: "integer",
				},
				extensions.Parameter{
					Name:     "SUFFIX",
					Required: true,
					Value:    "-hello-world",
				},
				extensions.Parameter{
					Name:     "REPLICAS",
				},
				extensions.Parameter{
					Name:     "STDIN",
					Type: "boolean",
				},
			},
			Objects: getObjects(t,
				getService("$(SUFFIX)", "$(PORT)"),
				getDeployment("$(SUFFIX)", "$(REPLICAS)", "$(STDIN)")),
		},
	}
	list, err := p.Process(ctx, &extensions.TemplateParameters{
		Name: "fake-name",
		ParameterValues: map[string]string{
			"PORT": "1080",
			"REPLICAS": "7",
			"STDIN": "true",
		},
	})
	assert.Nil(t, err)
	assert.NotEmpty(t, list)

	//expectedService := expectedService("-hello-world", 1080)
	//expectedDeployment := expectedDeployment("-hello-world", 7, true)
	//assert.Equal(t, 2, len(list.Items))

	//actualService := *(list.Items[0].(*v1.Service))
	//actualDeployment := *(list.Items[1].(*v1beta1.Deployment))
	//assert.EqualValues(t, expectedService, actualService)
	//assert.EqualValues(t, expectedDeployment, actualDeployment)
}
