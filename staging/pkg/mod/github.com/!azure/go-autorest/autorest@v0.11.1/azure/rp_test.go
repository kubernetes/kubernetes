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
	"context"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/mocks"
)

func TestDoRetryWithRegistration(t *testing.T) {
	client := mocks.NewSender()
	// first response, should retry because it is a transient error
	client.AppendResponse(mocks.NewResponseWithStatus("Internal server error", http.StatusInternalServerError))
	// response indicates the resource provider has not been registered
	client.AppendResponse(mocks.NewResponseWithBodyAndStatus(mocks.NewBody(`{
	"error":{
		"code":"MissingSubscriptionRegistration",
		"message":"The subscription registration is in 'Unregistered' state. The subscription must be registered to use namespace 'Microsoft.EventGrid'. See https://aka.ms/rps-not-found for how to register subscriptions.",
		"details":[
			{
				"code":"MissingSubscriptionRegistration",
				"target":"Microsoft.EventGrid",
				"message":"The subscription registration is in 'Unregistered' state. The subscription must be registered to use namespace 'Microsoft.EventGrid'. See https://aka.ms/rps-not-found for how to register subscriptions."
			}
		]
	}
}
`), http.StatusConflict, "MissingSubscriptionRegistration"))
	// first poll response, still not ready
	client.AppendResponse(mocks.NewResponseWithBodyAndStatus(mocks.NewBody(`{
	"registrationState": "Registering"
}
`), http.StatusOK, "200 OK"))
	// last poll response, respurce provider has been registered
	client.AppendResponse(mocks.NewResponseWithBodyAndStatus(mocks.NewBody(`{
	"registrationState": "Registered"
}
`), http.StatusOK, "200 OK"))
	// retry original request, response is successful
	client.AppendResponse(mocks.NewResponseWithStatus("200 OK", http.StatusOK))

	req := mocks.NewRequestForURL("https://lol/subscriptions/rofl")
	req.Body = mocks.NewBody("lolol")
	r, err := autorest.SendWithSender(client, req,
		DoRetryWithRegistration(autorest.Client{
			PollingDelay:    time.Second,
			PollingDuration: time.Second * 10,
			RetryAttempts:   5,
			RetryDuration:   time.Second,
			Sender:          client,
		}),
	)
	if err != nil {
		t.Fatalf("got error: %v", err)
	}

	autorest.Respond(r,
		autorest.ByDiscardingBody(),
		autorest.ByClosing(),
	)

	if r.StatusCode != http.StatusOK {
		t.Fatalf("azure: Sender#DoRetryWithRegistration -- Got: StatusCode %v; Want: StatusCode 200 OK", r.StatusCode)
	}
}

func TestDoRetrySkipRegistration(t *testing.T) {
	client := mocks.NewSender()
	// first response, should retry because it is a transient error
	client.AppendResponse(mocks.NewResponseWithStatus("Internal server error", http.StatusInternalServerError))
	// response indicates the resource provider has not been registered
	client.AppendResponse(mocks.NewResponseWithBodyAndStatus(mocks.NewBody(`{
	"error":{
		"code":"MissingSubscriptionRegistration",
		"message":"The subscription registration is in 'Unregistered' state. The subscription must be registered to use namespace 'Microsoft.EventGrid'. See https://aka.ms/rps-not-found for how to register subscriptions.",
		"details":[
			{
				"code":"MissingSubscriptionRegistration",
				"target":"Microsoft.EventGrid",
				"message":"The subscription registration is in 'Unregistered' state. The subscription must be registered to use namespace 'Microsoft.EventGrid'. See https://aka.ms/rps-not-found for how to register subscriptions."
			}
		]
	}
}`), http.StatusConflict, "MissingSubscriptionRegistration"))

	req := mocks.NewRequestForURL("https://lol/subscriptions/rofl")
	req.Body = mocks.NewBody("lolol")
	r, err := autorest.SendWithSender(client, req,
		DoRetryWithRegistration(autorest.Client{
			PollingDelay:                     time.Second,
			PollingDuration:                  time.Second * 10,
			RetryAttempts:                    5,
			RetryDuration:                    time.Second,
			Sender:                           client,
			SkipResourceProviderRegistration: true,
		}),
	)
	if err != nil {
		t.Fatalf("got error: %v", err)
	}

	autorest.Respond(r,
		autorest.ByDiscardingBody(),
		autorest.ByClosing(),
	)

	if r.StatusCode != http.StatusConflict {
		t.Fatalf("azure: Sender#DoRetryWithRegistration -- Got: StatusCode %v; Want: StatusCode 409 Conflict", r.StatusCode)
	}
}

func TestDoRetryWithRegistration_CanBeCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	delay := 5 * time.Second

	client := mocks.NewSender()
	client.AppendAndRepeatResponse(mocks.NewResponseWithStatus("Internal server error", http.StatusInternalServerError), 5)

	var wg sync.WaitGroup
	wg.Add(1)
	start := time.Now()
	end := time.Now()
	var err error

	go func() {
		req := mocks.NewRequestForURL("https://lol/subscriptions/rofl")
		req = req.WithContext(ctx)
		req.Body = mocks.NewBody("lolol")
		_, err = autorest.SendWithSender(client, req,
			DoRetryWithRegistration(autorest.Client{
				PollingDelay:                     time.Second,
				PollingDuration:                  delay,
				RetryAttempts:                    5,
				RetryDuration:                    time.Second,
				Sender:                           client,
				SkipResourceProviderRegistration: true,
			}),
		)
		end = time.Now()
		wg.Done()
	}()
	cancel()
	wg.Wait()
	time.Sleep(5 * time.Millisecond)
	if err == nil {
		t.Fatalf("azure: DoRetryWithRegistration didn't cancel")
	}
	if end.Sub(start) >= delay {
		t.Fatalf("azure: DoRetryWithRegistration failed to cancel")
	}
}
