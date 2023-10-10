package apivalidation

import (
	"k8s.io/apiserver/pkg/cel/apivalidation/defaulting"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

func (v *ValidationSchema) ApplyDefaults(val value.Value) {
	//!TODO: Move this to deserializer code?
	defaulting.Default(val, v.Schema)
}
