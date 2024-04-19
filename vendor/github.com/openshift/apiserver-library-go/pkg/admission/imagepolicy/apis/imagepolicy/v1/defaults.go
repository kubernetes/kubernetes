package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kapi "k8s.io/kubernetes/pkg/apis/core"
)

func SetDefaults_ImagePolicyConfig(obj *ImagePolicyConfig) {
	if obj == nil {
		return
	}

	if len(obj.ResolveImages) == 0 {
		obj.ResolveImages = Attempt
	}

	for i := range obj.ExecutionRules {
		if len(obj.ExecutionRules[i].OnResources) == 0 {
			obj.ExecutionRules[i].OnResources = []metav1.GroupResource{{Resource: "pods", Group: kapi.GroupName}}
		}
	}

	if obj.ResolutionRules == nil {
		obj.ResolutionRules = []ImageResolutionPolicyRule{
			{TargetResource: metav1.GroupResource{Group: "", Resource: "pods"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "", Resource: "replicationcontrollers"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "apps.openshift.io", Resource: "deploymentconfigs"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "apps", Resource: "daemonsets"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "apps", Resource: "deployments"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "apps", Resource: "statefulsets"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "apps", Resource: "replicasets"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "build.openshift.io", Resource: "builds"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "batch", Resource: "jobs"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "batch", Resource: "cronjobs"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "extensions", Resource: "daemonsets"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "extensions", Resource: "deployments"}, LocalNames: true},
			{TargetResource: metav1.GroupResource{Group: "extensions", Resource: "replicasets"}, LocalNames: true},
		}
		// default the resolution policy to the global default
		for i := range obj.ResolutionRules {
			if len(obj.ResolutionRules[i].Policy) != 0 {
				continue
			}
			obj.ResolutionRules[i].Policy = DoNotAttempt
			for _, rule := range obj.ExecutionRules {
				if executionRuleCoversResource(rule, obj.ResolutionRules[i].TargetResource) {
					obj.ResolutionRules[i].Policy = obj.ResolveImages
					break
				}
			}
		}
	} else {
		// default the resolution policy to the global default
		for i := range obj.ResolutionRules {
			if len(obj.ResolutionRules[i].Policy) != 0 {
				continue
			}
			obj.ResolutionRules[i].Policy = obj.ResolveImages
		}
	}

}

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	scheme.AddTypeDefaultingFunc(&ImagePolicyConfig{}, func(obj interface{}) { SetDefaults_ImagePolicyConfig(obj.(*ImagePolicyConfig)) })
	return nil
}

// executionRuleCoversResource returns true if gr is covered by rule.
func executionRuleCoversResource(rule ImageExecutionPolicyRule, gr metav1.GroupResource) bool {
	for _, target := range rule.OnResources {
		if target.Group == gr.Group && (target.Resource == gr.Resource || target.Resource == "*") {
			return true
		}
	}
	return false
}
