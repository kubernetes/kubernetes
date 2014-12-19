/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package validation

import (
	errs "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
)

func ValidateAccessToken(accessToken *api.OAuthAccessToken) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(accessToken.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("name", accessToken.Name))
	}
	if len(accessToken.ClientName) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("clientname", accessToken.ClientName))
	}
	if len(accessToken.UserName) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("username", accessToken.UserName))
	}
	if len(accessToken.UserUID) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("useruid", accessToken.UserUID))
	}
	allErrs = append(allErrs, validateLabels(accessToken.Labels)...)
	return allErrs
}

func ValidateAuthorizeToken(authorizeToken *api.OAuthAuthorizeToken) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(authorizeToken.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("name", authorizeToken.Name))
	}
	if len(authorizeToken.ClientName) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("clientname", authorizeToken.ClientName))
	}
	if len(authorizeToken.UserName) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("username", authorizeToken.UserName))
	}
	if len(authorizeToken.UserUID) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("useruid", authorizeToken.UserUID))
	}
	allErrs = append(allErrs, validateLabels(authorizeToken.Labels)...)
	return allErrs
}

func ValidateClient(client *api.OAuthClient) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(client.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("name", client.Name))
	}
	allErrs = append(allErrs, validateLabels(client.Labels)...)
	return allErrs
}

func ValidateClientAuthorization(clientAuthorization *api.OAuthClientAuthorization) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(clientAuthorization.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("name", clientAuthorization.Name))
	}
	if len(clientAuthorization.ClientName) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("clientname", clientAuthorization.ClientName))
	}
	if len(clientAuthorization.UserName) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("username", clientAuthorization.UserName))
	}
	if len(clientAuthorization.UserUID) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("useruid", clientAuthorization.UserUID))
	}
	allErrs = append(allErrs, validateLabels(clientAuthorization.Labels)...)
	return allErrs
}

func ValidateClientAuthorizationUpdate(newAuth *api.OAuthClientAuthorization, oldAuth *api.OAuthClientAuthorization) errs.ValidationErrorList {
	allErrs := ValidateClientAuthorization(newAuth)
	if oldAuth.Name != newAuth.Name {
		allErrs = append(allErrs, errs.NewFieldInvalid("name", newAuth.Name, "name cannot change during update"))
	}
	if oldAuth.ClientName != newAuth.ClientName {
		allErrs = append(allErrs, errs.NewFieldInvalid("clientName", newAuth.ClientName, "clientName cannot change during update"))
	}
	if oldAuth.UserName != newAuth.UserName {
		allErrs = append(allErrs, errs.NewFieldInvalid("userName", newAuth.UserName, "userName cannot change during update"))
	}
	if oldAuth.UserUID != newAuth.UserUID {
		allErrs = append(allErrs, errs.NewFieldInvalid("userUID", newAuth.UserUID, "userUID cannot change during update"))
	}
	allErrs = append(allErrs, validateLabels(newAuth.Labels)...)
	return allErrs
}

func validateLabels(labels map[string]string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	for k := range labels {
		if !util.IsDNS952Label(k) {
			allErrs = append(allErrs, errs.NewFieldNotSupported("label", k))
		}
	}
	return allErrs
}
