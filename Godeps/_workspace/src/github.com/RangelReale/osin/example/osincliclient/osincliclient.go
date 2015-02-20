package main

// Use github.com/RangelReale/osincli client to test
// Open url in browser:
// http://localhost:14001

import (
	"fmt"
	"github.com/RangelReale/osin"
	"github.com/RangelReale/osin/example"
	"github.com/RangelReale/osincli"
	"net/http"
)

func main() {
	// create http muxes
	serverhttp := http.NewServeMux()
	clienthttp := http.NewServeMux()

	// create server
	config := osin.NewServerConfig()
	sstorage := example.NewTestStorage()
	sstorage.SetClient("1234", &osin.DefaultClient{
		Id:          "1234",
		Secret:      "aabbccdd",
		RedirectUri: "http://localhost:14001/appauth",
	})
	server := osin.NewServer(config, sstorage)

	// create client
	cliconfig := &osincli.ClientConfig{
		ClientId:     "1234",
		ClientSecret: "aabbccdd",
		AuthorizeUrl: "http://localhost:14000/authorize",
		TokenUrl:     "http://localhost:14000/token",
		RedirectUrl:  "http://localhost:14001/appauth",
	}
	client, err := osincli.NewClient(cliconfig)
	if err != nil {
		panic(err)
	}

	// create a new request to generate the url
	areq := client.NewAuthorizeRequest(osincli.CODE)

	// SERVER

	// Authorization code endpoint
	serverhttp.HandleFunc("/authorize", func(w http.ResponseWriter, r *http.Request) {
		resp := server.NewResponse()
		defer resp.Close()

		if ar := server.HandleAuthorizeRequest(resp, r); ar != nil {
			if !example.HandleLoginPage(ar, w, r) {
				return
			}
			ar.Authorized = true
			server.FinishAuthorizeRequest(resp, r, ar)
		}
		if resp.IsError && resp.InternalError != nil {
			fmt.Printf("ERROR: %s\n", resp.InternalError)
		}
		osin.OutputJSON(resp, w, r)
	})

	// Access token endpoint
	serverhttp.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		resp := server.NewResponse()
		defer resp.Close()

		if ar := server.HandleAccessRequest(resp, r); ar != nil {
			ar.Authorized = true
			server.FinishAccessRequest(resp, r, ar)
		}
		if resp.IsError && resp.InternalError != nil {
			fmt.Printf("ERROR: %s\n", resp.InternalError)
		}
		osin.OutputJSON(resp, w, r)
	})

	// Information endpoint
	serverhttp.HandleFunc("/info", func(w http.ResponseWriter, r *http.Request) {
		resp := server.NewResponse()
		defer resp.Close()

		if ir := server.HandleInfoRequest(resp, r); ir != nil {
			server.FinishInfoRequest(resp, r, ir)
		}
		osin.OutputJSON(resp, w, r)
	})

	// CLIENT

	// Home
	clienthttp.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		u := areq.GetAuthorizeUrl()

		w.Write([]byte(fmt.Sprintf("<a href=\"%s\">Login</a>", u.String())))
	})

	// Auth endpoint
	clienthttp.HandleFunc("/appauth", func(w http.ResponseWriter, r *http.Request) {
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

	go http.ListenAndServe(":14001", clienthttp)
	http.ListenAndServe(":14000", serverhttp)
}
