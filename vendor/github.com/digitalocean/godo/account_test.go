package godo

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestAccountGet(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/account", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")

		response := `
		{ "account": {
			"droplet_limit": 25,
			"floating_ip_limit": 25,
			"email": "sammy@digitalocean.com",
			"uuid": "b6fr89dbf6d9156cace5f3c78dc9851d957381ef",
			"email_verified": true
			}
		}`

		fmt.Fprint(w, response)
	})

	acct, _, err := client.Account.Get()
	if err != nil {
		t.Errorf("Account.Get returned error: %v", err)
	}

	expected := &Account{DropletLimit: 25, FloatingIPLimit: 25, Email: "sammy@digitalocean.com",
		UUID: "b6fr89dbf6d9156cace5f3c78dc9851d957381ef", EmailVerified: true}
	if !reflect.DeepEqual(acct, expected) {
		t.Errorf("Account.Get returned %+v, expected %+v", acct, expected)
	}
}

func TestAccountString(t *testing.T) {
	acct := &Account{
		DropletLimit:    25,
		FloatingIPLimit: 25,
		Email:           "sammy@digitalocean.com",
		UUID:            "b6fr89dbf6d9156cace5f3c78dc9851d957381ef",
		EmailVerified:   true,
		Status:          "active",
		StatusMessage:   "message",
	}

	stringified := acct.String()
	expected := `godo.Account{DropletLimit:25, FloatingIPLimit:25, Email:"sammy@digitalocean.com", UUID:"b6fr89dbf6d9156cace5f3c78dc9851d957381ef", EmailVerified:true, Status:"active", StatusMessage:"message"}`
	if expected != stringified {
		t.Errorf("Account.String returned %+v, expected %+v", stringified, expected)
	}

}
