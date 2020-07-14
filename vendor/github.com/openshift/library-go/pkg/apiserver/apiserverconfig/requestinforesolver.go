package apiserverconfig

import (
	"k8s.io/apimachinery/pkg/util/sets"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func OpenshiftRequestInfoResolver() apirequest.RequestInfoResolver {
	// Default API request info factory
	requestInfoFactory := &apirequest.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}
	personalSARRequestInfoResolver := newPersonalSARRequestInfoResolver(requestInfoFactory)
	projectRequestInfoResolver := newProjectRequestInfoResolver(personalSARRequestInfoResolver)
	return projectRequestInfoResolver
}
