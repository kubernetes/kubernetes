package interfaces

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
)

type Result int

var (
	Allowed            Result = 1
	RequiresDefaulting Result = 2
	Forbidden          Result = 3
)

type ValidationResult struct {
	Result Result
	Errors field.ErrorList
}

func (r *ValidationResult) Add(result Result, errors ...*field.Error) {
	if r.Result < result {
		r.Result = result
	}
	r.Errors = append(r.Errors, errors...)
}

type PodValidatorDefaulter interface {
	ValidatePod(pod *api.Pod) (*ValidationResult, error)
	DefaultPod(pod *api.Pod) error
}

type ContainerValidatorDefaulter interface {
	ValidateContainer(pod *api.Pod, container *api.Container, effectiveSecurityContext *api.SecurityContext) (*ValidationResult, error)
	DefaultContainer(pod *api.Pod, container *api.Container) error
}
