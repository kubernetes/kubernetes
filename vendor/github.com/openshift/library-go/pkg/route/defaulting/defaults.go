package defaulting

import (
	v1 "github.com/openshift/api/route/v1"
)

// If adding or changing route defaults, updates may be required to
// pkg/router/controller/controller.go to ensure the routes generated from
// ingress resources will match routes created via the api.

func SetDefaults_RouteSpec(obj *v1.RouteSpec) {
	if len(obj.WildcardPolicy) == 0 {
		obj.WildcardPolicy = v1.WildcardPolicyNone
	}
}

func SetDefaults_RouteTargetReference(obj *v1.RouteTargetReference) {
	if len(obj.Kind) == 0 {
		obj.Kind = "Service"
	}
	if obj.Weight == nil {
		obj.Weight = new(int32)
		*obj.Weight = 100
	}
}

func SetDefaults_TLSConfig(obj *v1.TLSConfig) {
	if len(obj.Termination) == 0 && len(obj.DestinationCACertificate) == 0 {
		obj.Termination = v1.TLSTerminationEdge
	}
	switch obj.Termination {
	case v1.TLSTerminationType("Reencrypt"):
		obj.Termination = v1.TLSTerminationReencrypt
	case v1.TLSTerminationType("Edge"):
		obj.Termination = v1.TLSTerminationEdge
	case v1.TLSTerminationType("Passthrough"):
		obj.Termination = v1.TLSTerminationPassthrough
	}
}

func SetDefaults_RouteIngress(obj *v1.RouteIngress) {
	if len(obj.WildcardPolicy) == 0 {
		obj.WildcardPolicy = v1.WildcardPolicyNone
	}
}
