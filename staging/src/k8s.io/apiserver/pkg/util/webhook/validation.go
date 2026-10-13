/*
Copyright 2018 The Kubernetes Authors.

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

package webhook

import (
	"fmt"
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/client-go/transport"
)

func ValidateCABundle(fldPath *field.Path, caBundle []byte) field.ErrorList {
	var allErrors field.ErrorList
	_, err := transport.TLSConfigFor(&transport.Config{TLS: transport.TLSConfig{CAData: caBundle}})
	if err != nil {
		allErrors = append(allErrors, field.Invalid(fldPath, caBundle, err.Error()))
	}
	return allErrors
}

// ValidateWebhookURL validates webhook's URL.
func ValidateWebhookURL(fldPath *field.Path, URL string, forceHttps bool) field.ErrorList {
	var allErrors field.ErrorList
	const form = "; desired format: https://host[/path]"
	if u, err := url.Parse(URL); err != nil {
		allErrors = append(allErrors, field.Required(fldPath, "url must be a valid URL: "+err.Error()+form))
	} else {
		if forceHttps && u.Scheme != "https" {
			allErrors = append(allErrors, field.Invalid(fldPath, u.Scheme, "'https' is the only allowed URL scheme"+form))
		}
		if len(u.Host) == 0 {
			allErrors = append(allErrors, field.Invalid(fldPath, u.Host, "host must be specified"+form))
		}
		if u.User != nil {
			allErrors = append(allErrors, field.Invalid(fldPath, u.User.String(), "user information is not permitted in the URL"))
		}
		if len(u.Fragment) != 0 {
			allErrors = append(allErrors, field.Invalid(fldPath, u.Fragment, "fragments are not permitted in the URL"))
		}
		if len(u.RawQuery) != 0 {
			allErrors = append(allErrors, field.Invalid(fldPath, u.RawQuery, "query parameters are not permitted in the URL"))
		}
	}
	return allErrors
}

func ValidateWebhookService(fldPath *field.Path, namespace, name string, path *string, port int32) field.ErrorList {
	var allErrors field.ErrorList

	if len(name) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("name"), ""))
	}

	if len(namespace) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("namespace"), ""))
	}

	if errs := validation.IsValidPortNum(int(port)); errs != nil {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("port"), port, "port is not valid: "+strings.Join(errs, ", ")))
	}

	if path == nil {
		return allErrors
	}

	// TODO: replace below with url.Parse + verifying that host is empty?

	urlPath := *path
	if urlPath == "/" || len(urlPath) == 0 {
		return allErrors
	}
	if urlPath == "//" {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("path"), urlPath, "segment[0] may not be empty"))
		return allErrors
	}

	if !strings.HasPrefix(urlPath, "/") {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("path"), urlPath, "must start with a '/'"))
	}

	urlPathToCheck := urlPath[1:]
	if strings.HasSuffix(urlPathToCheck, "/") {
		urlPathToCheck = urlPathToCheck[:len(urlPathToCheck)-1]
	}
	steps := strings.Split(urlPathToCheck, "/")
	for i, step := range steps {
		if len(step) == 0 {
			allErrors = append(allErrors, field.Invalid(fldPath.Child("path"), urlPath, fmt.Sprintf("segment[%d] may not be empty", i)))
			continue
		}
		failures := validation.IsDNS1123Subdomain(step)
		for _, failure := range failures {
			allErrors = append(allErrors, field.Invalid(fldPath.Child("path"), urlPath, fmt.Sprintf("segment[%d]: %v", i, failure)))
		}
	}

	return allErrors
}
