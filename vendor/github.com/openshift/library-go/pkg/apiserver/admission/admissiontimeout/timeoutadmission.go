package admissiontimeout

import (
	"context"
	"fmt"
	"net/http"
	"runtime"
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

	type result struct {
		admissionErr error
		panicErr     error
	}
	// if a timeout occurs, we don't want the child goroutine to hang forever
	resultCh := make(chan result, 1)
	go func() {
		r := result{}
		// NOTE: panics don't cross goroutine boundaries, so we have to handle
		// the error here, we can't call utilruntime.HandleCrash here, then it
		// will cause the apiserver to crash.
		// We also need to make sure that the panic is propagated to the caller.
		// TODO: use the reusable panic handler once
		//  https://github.com/kubernetes/kubernetes/pull/115564 merges.
		defer func() {
			if err := recover(); err != nil {
				r.panicErr = fmt.Errorf("admission panic'd: %v", stack(err))
				utilruntime.HandleError(r.panicErr)
			}
			resultCh <- r
		}()

		r.admissionErr = mutatingHandler.Admit(ctx, a, o)
	}()

	select {
	case r := <-resultCh:
		if r.panicErr != nil {
			// this panic will propagate to net/http
			panic(r.panicErr.(interface{}))
		}
		return r.admissionErr
	case <-time.After(p.timeout):
		return errors.NewInternalError(fmt.Errorf("admission plugin %q failed to complete mutation in %v", p.name, p.timeout))
	}
}

func (p pluginHandlerWithTimeout) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	validatingHandler, ok := p.admissionPlugin.(admission.ValidationInterface)
	if !ok {
		return nil
	}

	type result struct {
		admissionErr error
		panicErr     error
	}
	// if a timeout occurs, we don't want the child goroutine to hang forever
	resultCh := make(chan result, 1)
	go func() {
		r := result{}
		// NOTE: panics don't cross goroutine boundaries, so we have to handle
		// the error here, we can't call utilruntime.HandleCrash here, then it
		// will cause the apiserver to crash.
		// We also need to make sure that the panic is propagated to the caller.
		// TODO: use the reusable panic handler once
		//  https://github.com/kubernetes/kubernetes/pull/115564 merges.
		defer func() {
			if err := recover(); err != nil {
				r.panicErr = fmt.Errorf("admission panic'd: %v", stack(err))
				utilruntime.HandleError(r.panicErr)
			}
			resultCh <- r
		}()

		r.admissionErr = validatingHandler.Validate(ctx, a, o)
	}()

	select {
	case r := <-resultCh:
		if r.panicErr != nil {
			// this panic will propagate to net/http
			panic(r.panicErr.(interface{}))
		}
		return r.admissionErr
	case <-time.After(p.timeout):
		return errors.NewInternalError(fmt.Errorf("admission plugin %q failed to complete validation in %v", p.name, p.timeout))
	}
}

func stack(recovered interface{}) interface{} {
	// do not wrap the sentinel ErrAbortHandler panic value
	if recovered == http.ErrAbortHandler {
		return recovered
	}

	// Same as stdlib http server code. Manually allocate stack
	// trace buffer size to prevent excessively large logs
	const size = 64 << 10
	buf := make([]byte, size)
	buf = buf[:runtime.Stack(buf, false)]
	return fmt.Sprintf("%v\n%s", recovered, buf)
}
