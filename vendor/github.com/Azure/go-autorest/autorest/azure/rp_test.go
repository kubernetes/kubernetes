package azure

import (
	"net/http"
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
