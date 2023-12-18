package admissiontimeout

import (
	"time"

	"k8s.io/apiserver/pkg/admission"
)

// AdmissionTimeout provides a decorator that will fail an admission plugin after a certain amount of time
//
// DEPRECATED: use the context of the admission handler instead.
type AdmissionTimeout struct {
	Timeout time.Duration
}

func (d AdmissionTimeout) WithTimeout(admissionPlugin admission.Interface, name string) admission.Interface {
	return pluginHandlerWithTimeout{
		name:            name,
		admissionPlugin: admissionPlugin,
		timeout:         d.Timeout,
	}
}
