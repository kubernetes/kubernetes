// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

package azure

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Azure/go-autorest/autorest"
)

// DoRetryWithRegistration tries to register the resource provider in case it is unregistered.
// It also handles request retries
func DoRetryWithRegistration(client autorest.Client) autorest.SendDecorator {
	return func(s autorest.Sender) autorest.Sender {
		return autorest.SenderFunc(func(r *http.Request) (resp *http.Response, err error) {
			rr := autorest.NewRetriableRequest(r)
			for currentAttempt := 0; currentAttempt < client.RetryAttempts; currentAttempt++ {
				err = rr.Prepare()
				if err != nil {
					return resp, err
				}

				resp, err = autorest.SendWithSender(s, rr.Request(),
					autorest.DoRetryForStatusCodes(client.RetryAttempts, client.RetryDuration, autorest.StatusCodesForRetry...),
				)
				if err != nil {
					return resp, err
				}

				if resp.StatusCode != http.StatusConflict || client.SkipResourceProviderRegistration {
					return resp, err
				}
				var re RequestError
				err = autorest.Respond(
					resp,
					autorest.ByUnmarshallingJSON(&re),
				)
				if err != nil {
					return resp, err
				}
				err = re

				if re.ServiceError != nil && re.ServiceError.Code == "MissingSubscriptionRegistration" {
					regErr := register(client, r, re)
					if regErr != nil {
						return resp, fmt.Errorf("failed auto registering Resource Provider: %s. Original error: %s", regErr, err)
					}
				}
			}
			return resp, err
		})
	}
}

func getProvider(re RequestError) (string, error) {
	if re.ServiceError != nil && len(re.ServiceError.Details) > 0 {
		return re.ServiceError.Details[0]["target"].(string), nil
	}
	return "", errors.New("provider was not found in the response")
}

func register(client autorest.Client, originalReq *http.Request, re RequestError) error {
	subID := getSubscription(originalReq.URL.Path)
	if subID == "" {
		return errors.New("missing parameter subscriptionID to register resource provider")
	}
	providerName, err := getProvider(re)
	if err != nil {
		return fmt.Errorf("missing parameter provider to register resource provider: %s", err)
	}
	newURL := url.URL{
		Scheme: originalReq.URL.Scheme,
		Host:   originalReq.URL.Host,
	}

	// taken from the resources SDK
	// with almost identical code, this sections are easier to mantain
	// It is also not a good idea to import the SDK here
	// https://github.com/Azure/azure-sdk-for-go/blob/9f366792afa3e0ddaecdc860e793ba9d75e76c27/arm/resources/resources/providers.go#L252
	pathParameters := map[string]interface{}{
		"resourceProviderNamespace": autorest.Encode("path", providerName),
		"subscriptionId":            autorest.Encode("path", subID),
	}

	const APIVersion = "2016-09-01"
	queryParameters := map[string]interface{}{
		"api-version": APIVersion,
	}

	preparer := autorest.CreatePreparer(
		autorest.AsPost(),
		autorest.WithBaseURL(newURL.String()),
		autorest.WithPathParameters("/subscriptions/{subscriptionId}/providers/{resourceProviderNamespace}/register", pathParameters),
		autorest.WithQueryParameters(queryParameters),
	)

	req, err := preparer.Prepare(&http.Request{})
	if err != nil {
		return err
	}
	req = req.WithContext(originalReq.Context())

	resp, err := autorest.SendWithSender(client, req,
		autorest.DoRetryForStatusCodes(client.RetryAttempts, client.RetryDuration, autorest.StatusCodesForRetry...),
	)
	if err != nil {
		return err
	}

	type Provider struct {
		RegistrationState *string `json:"registrationState,omitempty"`
	}
	var provider Provider

	err = autorest.Respond(
		resp,
		WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&provider),
		autorest.ByClosing(),
	)
	if err != nil {
		return err
	}

	// poll for registered provisioning state
	now := time.Now()
	for err == nil && time.Since(now) < client.PollingDuration {
		// taken from the resources SDK
		// https://github.com/Azure/azure-sdk-for-go/blob/9f366792afa3e0ddaecdc860e793ba9d75e76c27/arm/resources/resources/providers.go#L45
		preparer := autorest.CreatePreparer(
			autorest.AsGet(),
			autorest.WithBaseURL(newURL.String()),
			autorest.WithPathParameters("/subscriptions/{subscriptionId}/providers/{resourceProviderNamespace}", pathParameters),
			autorest.WithQueryParameters(queryParameters),
		)
		req, err = preparer.Prepare(&http.Request{})
		if err != nil {
			return err
		}
		req = req.WithContext(originalReq.Context())

		resp, err := autorest.SendWithSender(client, req,
			autorest.DoRetryForStatusCodes(client.RetryAttempts, client.RetryDuration, autorest.StatusCodesForRetry...),
		)
		if err != nil {
			return err
		}

		err = autorest.Respond(
			resp,
			WithErrorUnlessStatusCode(http.StatusOK),
			autorest.ByUnmarshallingJSON(&provider),
			autorest.ByClosing(),
		)
		if err != nil {
			return err
		}

		if provider.RegistrationState != nil &&
			*provider.RegistrationState == "Registered" {
			break
		}

		delayed := autorest.DelayWithRetryAfter(resp, originalReq.Context().Done())
		if !delayed && !autorest.DelayForBackoff(client.PollingDelay, 0, originalReq.Context().Done()) {
			return originalReq.Context().Err()
		}
	}
	if !(time.Since(now) < client.PollingDuration) {
		return errors.New("polling for resource provider registration has exceeded the polling duration")
	}
	return err
}

func getSubscription(path string) string {
	parts := strings.Split(path, "/")
	for i, v := range parts {
		if v == "subscriptions" && (i+1) < len(parts) {
			return parts[i+1]
		}
	}
	return ""
}
