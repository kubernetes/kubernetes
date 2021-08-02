package admissiontimeout

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
)

type pluginHandlerWithTimeout struct {
	name            string
	admissionPlugin admission.Interface
	timeout         time.Duration
}

var _ admission.ValidationInterface = &pluginHandlerWithTimeout{}
var _ admission.MutationInterface = &pluginHandlerWithTimeout{}

func (p pluginHandlerWithTimeout) Handles(operation admission.Operation) bool {
	return p.admissionPlugin.Handles(operation)
}

func (p pluginHandlerWithTimeout) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	mutatingHandler, ok := p.admissionPlugin.(admission.MutationInterface)
	if !ok {
		return nil
	}

	admissionDone := make(chan struct{})
	admissionErr := fmt.Errorf("default to mutation error")
	go func() {
		defer utilruntime.HandleCrash()
		defer close(admissionDone)
		admissionErr = mutatingHandler.Admit(ctx, a, o)
	}()

	select {
	case <-admissionDone:
		return admissionErr
	case <-time.After(p.timeout):
		return errors.NewInternalError(fmt.Errorf("admission plugin %q failed to complete mutation in %v", p.name, p.timeout))
	}
}

func (p pluginHandlerWithTimeout) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	validatingHandler, ok := p.admissionPlugin.(admission.ValidationInterface)
	if !ok {
		return nil
	}

	admissionDone := make(chan struct{})
	admissionErr := fmt.Errorf("default to validation error")
	go func() {
		defer utilruntime.HandleCrash()
		defer close(admissionDone)
		admissionErr = validatingHandler.Validate(ctx, a, o)
	}()

	select {
	case <-admissionDone:
		return admissionErr
	case <-time.After(p.timeout):
		return errors.NewInternalError(fmt.Errorf("admission plugin %q failed to complete validation in %v", p.name, p.timeout))
	}
}
