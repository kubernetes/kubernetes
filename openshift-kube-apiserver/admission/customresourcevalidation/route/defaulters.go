package route

import (
	routev1 "github.com/openshift/api/route/v1"
	"github.com/openshift/library-go/pkg/route/defaulting"
)

// Defaulters defined in github.com/openshift/library-go/pkg/route/defaulting are not recongized by
// codegen (make update). This file MUST contain duplicates of each defaulter function defined in
// library-go, with the body of each function defined here delegating to its library-go
// counterpart. Missing or extra defaulters here will introduce differences between Route as a CRD
// (MicroShift) and Route as an aggregated API of openshift-apiserver.

func SetDefaults_RouteSpec(obj *routev1.RouteSpec) {
	defaulting.SetDefaults_RouteSpec(obj)
}

func SetDefaults_RouteTargetReference(obj *routev1.RouteTargetReference) {
	defaulting.SetDefaults_RouteTargetReference(obj)
}

func SetDefaults_TLSConfig(obj *routev1.TLSConfig) {
	defaulting.SetDefaults_TLSConfig(obj)
}

func SetDefaults_RouteIngress(obj *routev1.RouteIngress) {
	defaulting.SetDefaults_RouteIngress(obj)
}
