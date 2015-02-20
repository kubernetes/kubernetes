package main

import (
	"github.com/RangelReale/osincli"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
)

func main() {
	config := &osincli.ClientConfig{
		ClientId:                 "xxxxxxxxxxxx.apps.googleusercontent.com",
		ClientSecret:             "secret",
		AuthorizeUrl:             "https://accounts.google.com/o/oauth2/auth",
		TokenUrl:                 "https://accounts.google.com/o/oauth2/token",
		RedirectUrl:              "http://localhost:14001/appauth",
		ErrorsInStatusCode:       true,
		SendClientSecretInParams: true,
		Scope: "https://www.googleapis.com/auth/plus.login",
	}
	client, err := osincli.NewClient(config)
	if err != nil {
		panic(err)
	}

	// create a new request to generate the url
	areq := client.NewAuthorizeRequest(osincli.CODE)
	areq.CustomParameters["access_type"] = "online"
	areq.CustomParameters["approval_prompt"] = "auto"

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
			w.Write([]byte(fmt.Sprintf("Access token URL: %s\n\n", u2.String())))

			// exchange the authorize token for the access token
			ad, err := treq.GetToken()
			if err == nil {
				w.Write([]byte(fmt.Sprintf("Access token: %+v\n\n", ad)))

				AccessPlusApi(ad, w)
			} else {
				w.Write([]byte(fmt.Sprintf("ERROR: %s\n", err)))
			}
		} else {
			w.Write([]byte(fmt.Sprintf("ERROR: %s\n", err)))
		}
	})

	http.ListenAndServe(":14001", nil)
}

func AccessPlusApi(ad *osincli.AccessData, w http.ResponseWriter) {
	u, err := url.Parse("https://www.googleapis.com/plus/v1/people/me")
	if err != nil {
		panic(err)
	}
	uq := u.Query()
	uq.Set("access_token", ad.AccessToken)
	u.RawQuery = uq.Encode()

	w.Write([]byte(fmt.Sprintf("Accessing: %s\n\n", u.String())))

	resp, err := http.Get(u.String())
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		w.Write([]byte(fmt.Sprintf("Invalid status code: %d (%s)\n", resp.StatusCode, resp.Status)))
		return
	}

	if b, err := ioutil.ReadAll(resp.Body); err == nil {
		w.Write(b)
	} else {
		w.Write([]byte(err.Error()))
	}

}
