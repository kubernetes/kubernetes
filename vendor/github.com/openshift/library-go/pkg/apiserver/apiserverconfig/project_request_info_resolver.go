package apiserverconfig

import (
	"net/http"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"

	projectv1 "github.com/openshift/api/project/v1"
)

type projectRequestInfoResolver struct {
	// infoFactory is used to determine info for the request
	infoFactory apirequest.RequestInfoResolver
}

func newProjectRequestInfoResolver(infoFactory apirequest.RequestInfoResolver) apirequest.RequestInfoResolver {
	return &projectRequestInfoResolver{
		infoFactory: infoFactory,
	}
}

func (a *projectRequestInfoResolver) NewRequestInfo(req *http.Request) (*apirequest.RequestInfo, error) {
	requestInfo, err := a.infoFactory.NewRequestInfo(req)
	if err != nil {
		return requestInfo, err
	}

	// if the resource is projects, we need to set the namespace to the value of the name.
	if (len(requestInfo.APIGroup) == 0 || requestInfo.APIGroup == projectv1.GroupName) && requestInfo.Resource == "projects" && len(requestInfo.Name) > 0 {
		requestInfo.Namespace = requestInfo.Name
	}

	return requestInfo, nil
}
