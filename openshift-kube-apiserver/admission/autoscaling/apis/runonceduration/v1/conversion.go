package v1

import (
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"

	internal "k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/runonceduration"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	err := scheme.AddConversionFunc((*RunOnceDurationConfig)(nil), (*internal.RunOnceDurationConfig)(nil), func(a, b interface{}, scope conversion.Scope) error {
		in := a.(*RunOnceDurationConfig)
		out := b.(*internal.RunOnceDurationConfig)
		out.ActiveDeadlineSecondsLimit = in.ActiveDeadlineSecondsOverride
		return nil
	})
	if err != nil {
		return err
	}
	return scheme.AddConversionFunc((*internal.RunOnceDurationConfig)(nil), (*RunOnceDurationConfig)(nil), func(a, b interface{}, scope conversion.Scope) error {
		in := a.(*internal.RunOnceDurationConfig)
		out := b.(*RunOnceDurationConfig)
		out.ActiveDeadlineSecondsOverride = in.ActiveDeadlineSecondsLimit
		return nil
	})
}
