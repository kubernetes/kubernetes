package apiserverconfig

import (
	"bytes"
	"io"
	"net/http"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/endpoints/request"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"

	authorizationv1 "github.com/openshift/api/authorization/v1"
)

type personalSARRequestInfoResolver struct {
	// infoFactory is used to determine info for the request
	infoFactory apirequest.RequestInfoResolver
}

func newPersonalSARRequestInfoResolver(infoFactory apirequest.RequestInfoResolver) apirequest.RequestInfoResolver {
	return &personalSARRequestInfoResolver{
		infoFactory: infoFactory,
	}
}

func (a *personalSARRequestInfoResolver) NewRequestInfo(req *http.Request) (*request.RequestInfo, error) {
	requestInfo, err := a.infoFactory.NewRequestInfo(req)
	if err != nil {
		return requestInfo, err
	}

	// only match SAR and LSAR requests for personal review
	switch {
	case !requestInfo.IsResourceRequest:
		return requestInfo, nil

	case len(requestInfo.APIGroup) != 0 && requestInfo.APIGroup != "authorization.openshift.io":
		return requestInfo, nil

	case len(requestInfo.Subresource) != 0:
		return requestInfo, nil

	case requestInfo.Verb != "create":
		return requestInfo, nil

	case requestInfo.Resource != "subjectaccessreviews" && requestInfo.Resource != "localsubjectaccessreviews":
		return requestInfo, nil
	}

	// at this point we're probably running a SAR or LSAR.  Decode the body and check.  This is expensive.
	isSelfSAR, err := isPersonalAccessReviewFromRequest(req, requestInfo)
	if err != nil {
		return nil, err
	}
	if !isSelfSAR {
		return requestInfo, nil
	}

	// if we do have a self-SAR, rewrite the requestInfo to indicate this is a selfsubjectaccessreviews.authorization.k8s.io request
	requestInfo.APIGroup = "authorization.k8s.io"
	requestInfo.Resource = "selfsubjectaccessreviews"

	return requestInfo, nil
}

// isPersonalAccessReviewFromRequest this variant handles the case where we have an httpRequest
func isPersonalAccessReviewFromRequest(req *http.Request, requestInfo *request.RequestInfo) (bool, error) {
	// TODO once we're integrated with the api installer, we should have direct access to the deserialized content
	// for now, this only happens on subjectaccessreviews with a personal check, pay the double retrieve and decode cost
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return false, err
	}
	req.Body = io.NopCloser(bytes.NewBuffer(body))

	defaultGVK := schema.GroupVersionKind{Version: requestInfo.APIVersion, Group: requestInfo.APIGroup}
	switch requestInfo.Resource {
	case "subjectaccessreviews":
		defaultGVK.Kind = "SubjectAccessReview"
	case "localsubjectaccessreviews":
		defaultGVK.Kind = "LocalSubjectAccessReview"
	}

	obj, _, err := sarCodecFactory.UniversalDeserializer().Decode(body, &defaultGVK, nil)
	if err != nil {
		return false, err
	}
	switch castObj := obj.(type) {
	case *authorizationv1.SubjectAccessReview:
		return IsPersonalAccessReviewFromSAR(castObj), nil

	case *authorizationv1.LocalSubjectAccessReview:
		return isPersonalAccessReviewFromLocalSAR(castObj), nil

	default:
		return false, nil
	}
}

// IsPersonalAccessReviewFromSAR this variant handles the case where we have an SAR
func IsPersonalAccessReviewFromSAR(sar *authorizationv1.SubjectAccessReview) bool {
	if len(sar.User) == 0 && len(sar.GroupsSlice) == 0 {
		return true
	}

	return false
}

// isPersonalAccessReviewFromLocalSAR this variant handles the case where we have a local SAR
func isPersonalAccessReviewFromLocalSAR(sar *authorizationv1.LocalSubjectAccessReview) bool {
	if len(sar.User) == 0 && len(sar.GroupsSlice) == 0 {
		return true
	}

	return false
}

var (
	sarScheme       = runtime.NewScheme()
	sarCodecFactory = serializer.NewCodecFactory(sarScheme)
)

func init() {
	utilruntime.Must(authorizationv1.Install(sarScheme))
	utilruntime.Must(authorizationv1.DeprecatedInstallWithoutGroup(sarScheme))
}
