// To test with the osin samples, change the RedirectUri to "http://localhost:14001/appauth"
// in osin/examples/teststorage.to
package main

import (
	"fmt"
	"github.com/RangelReale/osincli"
	"net/http"
)

func main() {
	config := &osincli.ClientConfig{
		ClientId:     "1234",
		ClientSecret: "aabbccdd",
		AuthorizeUrl: "http://localhost:14000/authorize",
		TokenUrl:     "http://localhost:14000/token",
		RedirectUrl:  "http://localhost:14001/appauth",
	}
	client, err := osincli.NewClient(config)
	if err != nil {
		panic(err)
	}

	// create a new request to generate the url
	areq := client.NewAuthorizeRequest(osincli.CODE)

	// Home
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		u := areq.GetAuthorizeUrl()

		w.Write([]byte(fmt.Sprintf("<a href=\"%s\">Login</a>", u.String())))
	})

	// Auth endpoint
	http.HandleFunc("/appauth", func(w http.ResponseWriter, r *http.Request) {
		// parse a token request
		if areqdata, err := areq.HandleRequest(r); err == nil {
			treq := client.NewAccessRequest(osincli.AUTHORIZATION_CODE, areqdata)

			// show access request url (for debugging only)
			u2 := treq.GetTokenUrl()
			w.Write([]byte(fmt.Sprintf("Access token URL: %s\n", u2.String())))

			// exchange the authorize token for the access token
			ad, err := treq.GetToken()
			if err == nil {
				w.Write([]byte(fmt.Sprintf("Access token: %+v\n", ad)))
			} else {
				w.Write([]byte(fmt.Sprintf("ERROR: %s\n", err)))
			}
		} else {
			w.Write([]byte(fmt.Sprintf("ERROR: %s\n", err)))
		}
	})

	http.ListenAndServe(":14001", nil)
}
