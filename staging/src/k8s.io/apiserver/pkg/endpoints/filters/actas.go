package filters

import (
	"errors"
	"fmt"
	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/httplog"
	"k8s.io/klog/v2"
	"net/http"
	"strings"
)

const (
	ActAsUserHeader        = "ActAs-User"
	ActAsGroupHeader       = "ActAs-Group"
	ActAsExtraHeaderPrefix = "ActAs-Extra"
	ActAsUID               = "ActAs-Uid"
)

func WithActAs(handler http.Handler, a authorizer.Authorizer, s runtime.NegotiatedSerializer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		actAsRequests, err := buildActAsRequests(req.Header)
		if err != nil {
			klog.V(4).Infof("%v", err)
			responsewriters.InternalError(w, req, err)
			return
		}
		if len(actAsRequests) == 0 {
			handler.ServeHTTP(w, req)
			return
		}

		ctx := req.Context()
		requestor, exists := request.UserFrom(ctx)
		if !exists {
			responsewriters.InternalError(w, req, errors.New("no user found for request"))
			return
		}

		attributes, err := GetAuthorizerAttributes(ctx)
		if err != nil {
			responsewriters.InternalError(w, req, err)
			return
		}

		// if groups are not specified, then we need to look them up differently depending on the type of user
		// if they are specified, then they are the authority (including the inclusion of system:authenticated/system:unauthenticated groups)
		groupsSpecified := len(req.Header[ActAsGroupHeader]) > 0

		// make sure we're allowed to actas each thing we're requesting.  While we're iterating through, start building username
		// and group information
		username := ""
		groups := []string{}
		userExtra := map[string][]string{}
		uid := ""
		for _, actAsRequest := range actAsRequests {
			gvk := actAsRequest.GetObjectKind().GroupVersionKind()
			actingAsAttributes := &authorizer.AttributesRecord{
				User:            requestor,
				Verb:            "actas",
				APIGroup:        gvk.Group,
				APIVersion:      gvk.Version,
				Namespace:       actAsRequest.Namespace,
				Name:            actAsRequest.Name,
				ResourceRequest: true,
			}

			switch gvk.GroupKind() {
			case v1.SchemeGroupVersion.WithKind("ServiceAccount").GroupKind():
				actingAsAttributes.Resource = "serviceaccounts"
				username = serviceaccount.MakeUsername(actAsRequest.Namespace, actAsRequest.Name)
				if !groupsSpecified {
					// if groups aren't specified for a service account, we know the groups because its a fixed mapping.  Add them
					groups = serviceaccount.MakeGroupNames(actAsRequest.Namespace)
				}

			case v1.SchemeGroupVersion.WithKind("User").GroupKind():
				actingAsAttributes.Resource = "users"
				username = actAsRequest.Name

			case v1.SchemeGroupVersion.WithKind("Group").GroupKind():
				actingAsAttributes.Resource = "groups"
				groups = append(groups, actAsRequest.Name)

			case authenticationv1.SchemeGroupVersion.WithKind("UserExtra").GroupKind():
				extraKey := actAsRequest.FieldPath
				extraValue := actAsRequest.Name
				actingAsAttributes.Resource = "userextras"
				actingAsAttributes.Subresource = extraKey
				userExtra[extraKey] = append(userExtra[extraKey], extraValue)

			case authenticationv1.SchemeGroupVersion.WithKind("UID").GroupKind():
				uid = string(actAsRequest.Name)
				actingAsAttributes.Resource = "uids"

			default:
				klog.V(4).InfoS("unknown actas request type", "request", actAsRequest)
				responsewriters.Forbidden(ctx, actingAsAttributes, w, req, fmt.Sprintf("unknown actas request type: %v", actAsRequest), s)
				return
			}

			decision, reason, err := a.Authorize(ctx, actingAsAttributes)
			if err != nil || decision != authorizer.DecisionAllow {
				klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", reason, "err", err)
				responsewriters.Forbidden(ctx, actingAsAttributes, w, req, reason, s)
				return
			}
		}

		authorized, reason, err := a.Authorize(ctx, attributes)
		if err != nil || authorized != authorizer.DecisionAllow {
			klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", reason, "err", err)
			responsewriters.Forbidden(ctx, attributes, w, req, reason, s)
			return
		}

		if username != user.Anonymous {
			// When acting as a non-anonymous user, include the 'system:authenticated' group
			// in the acted user info:
			// - if no groups were specified
			// - if a group has been specified other than 'system:authenticated'
			//
			// If 'system:unauthenticated' group has been specified we should not include
			// the 'system:authenticated' group.
			addAuthenticated := true
			for _, group := range groups {
				if group == user.AllAuthenticated || group == user.AllUnauthenticated {
					addAuthenticated = false
					break
				}
			}

			if addAuthenticated {
				groups = append(groups, user.AllAuthenticated)
			}
		} else {
			addUnauthenticated := true
			for _, group := range groups {
				if group == user.AllUnauthenticated {
					addUnauthenticated = false
					break
				}
			}

			if addUnauthenticated {
				groups = append(groups, user.AllUnauthenticated)
			}
		}

		newUser := &user.DefaultInfo{
			Name:   username,
			Groups: groups,
			Extra:  userExtra,
			UID:    uid,
		}
		req = req.WithContext(request.WithUser(ctx, newUser))

		oldUser, _ := request.UserFrom(ctx)
		httplog.LogOf(req, w).Addf("%v is acting as %v", userString(oldUser), userString(newUser))

		ae := audit.AuditEventFrom(ctx)
		audit.LogImpersonatedUser(ae, newUser)

		// clear all the actas headers from the request
		req.Header.Del(ActAsUserHeader)
		req.Header.Del(ActAsGroupHeader)
		req.Header.Del(ActAsUID)
		for headerName := range req.Header {
			if strings.HasPrefix(headerName, ActAsExtraHeaderPrefix) {
				req.Header.Del(headerName)
			}
		}

		handler.ServeHTTP(w, req)
	})
}

func buildActAsRequests(headers http.Header) ([]v1.ObjectReference, error) {
	actAsRequests := []v1.ObjectReference{}

	requestedUser := headers.Get(ActAsUserHeader)
	hasUser := len(requestedUser) > 0
	if hasUser {
		if namespace, name, err := serviceaccount.SplitUsername(requestedUser); err == nil {
			actAsRequests = append(actAsRequests, v1.ObjectReference{Kind: "ServiceAccount", Namespace: namespace, Name: name})
		} else {
			actAsRequests = append(actAsRequests, v1.ObjectReference{Kind: "User", Name: requestedUser})
		}
	}

	hasGroups := false
	for _, group := range headers[ActAsGroupHeader] {
		hasGroups = true
		actAsRequests = append(actAsRequests, v1.ObjectReference{Kind: "Group", Name: group})
	}

	hasUserExtra := false
	for headerName, values := range headers {
		if !strings.HasPrefix(headerName, ActAsExtraHeaderPrefix) {
			continue
		}

		hasUserExtra = true
		extraKey := unescapeExtraKey(strings.ToLower(headerName[len(ActAsExtraHeaderPrefix):]))

		// make a separate request for each extra value they're trying to set
		for _, value := range values {
			actAsRequests = append(actAsRequests,
				v1.ObjectReference{
					Kind: "UserExtra",
					// we only parse out a group above, but the parsing will fail if there isn't SOME version
					// using the internal version will help us fail if anyone starts using it
					APIVersion: authenticationv1.SchemeGroupVersion.String(),
					Name:       value,
					// ObjectReference doesn't have a subresource field.  FieldPath is close and available, so we'll use that
					// TODO fight the good fight for ObjectReference to refer to resources and subresources
					FieldPath: extraKey,
				})
		}
	}

	requestedUID := headers.Get(ActAsUID)
	hasUID := len(requestedUID) > 0
	if hasUID {
		actAsRequests = append(actAsRequests, v1.ObjectReference{
			Kind:       "UID",
			Name:       requestedUID,
			APIVersion: authenticationv1.SchemeGroupVersion.String(),
		})
	}

	if (hasGroups || hasUserExtra || hasUID) && !hasUser {
		return nil, fmt.Errorf("requested %v without acting as a user", actAsRequests)
	}

	return actAsRequests, nil
}
