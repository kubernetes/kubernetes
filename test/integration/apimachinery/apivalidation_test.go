package apimachinery

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/cel/apivalidation"
	"k8s.io/apiserver/pkg/cel/openapi"
	openapiresolver "k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
)

func TestSchemaValidation(t *testing.T) {
	// My dude what a cool validator...can you validate a config map tho?
	schemaResolver := openapiresolver.NewDefinitionsSchemaResolver(scheme.Scheme, generatedopenapi.GetOpenAPIDefinitions)
	configMapSchema, err := schemaResolver.ResolveSchema(
		schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
	)
	require.NoError(t, err)
	validationSchema := apivalidation.NewValidationSchema(&openapi.Schema{Schema: configMapSchema})
	options := apivalidation.ValidationOptions{
		Ratcheting: true,
		Registry:   strfmt.Default,
	}
	value := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			CreationTimestamp: metav1.Now(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					ResizePolicy: []v1.ContainerResizePolicy{
						{
							RestartPolicy: v1.ResourceResizeRestartPolicy("INVALID HAHA"),
						},
					},
				},
			},
		},
	}
	scheme.Scheme.Default(value)

	errs := validationSchema.Validate(value, options)
	assert.Empty(t, errs)
}
