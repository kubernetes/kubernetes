OSIN CLIENT
===========

Golang OAuth2 client library
----------------------------

OSINCLI is an OAuth2 client library for the Go language, as specified at
http://tools.ietf.org/html/rfc6749.

Using it, you can access an OAuth2 authenticated service.

The library follows the RFC recommendations, but allows some differences, like passing the client secret in the url instead of using the Authorization header.

### Example

	import "github.com/RangelReale/osincli"

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

			// exchange the authorize token for the access token
			ad, err := treq.GetToken()
			if err == nil {
				w.Write([]byte(fmt.Sprintf("Access token: %+v\n\n", ad)))
				
				// use the token in ad.AccessToken
			} else {
				w.Write([]byte(fmt.Sprintf("ERROR: %s\n", err)))
			}
		} else {
			w.Write([]byte(fmt.Sprintf("ERROR: %s\n", err)))
		}
	})

	http.ListenAndServe(":14001", nil)


Load your web browser at:

	http://localhost:14001

### License

The code is licensed using "New BSD" license.

### Author

Rangel Reale
