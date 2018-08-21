package options

import "k8s.io/apiserver/pkg/admission"

var RegisterAllAdmissionPlugins = registerAllAdmissionPlugins

var DefaultOffAdmissionPlugins = defaultOffAdmissionPlugins

var Decorators = []admission.Decorator{}
