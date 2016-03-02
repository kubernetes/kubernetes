// +build ignore

/*
 Example demonstrating how to wrap an application which is unaware of
 authenticated requests with a "pass-through" authentication

 Build with:

 go build wrapped.go
*/

package main

import (
	auth ".."
	"fmt"
	"net/http"
)

func Secret(user, realm string) string {
	if user == "john" {
		// password is "hello"
		return "$1$dlPL2MqE$oQmn16q49SqdmhenQuNgs1"
	}
	return ""
}

func regular_handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "<html><body><h1>This application is unaware of authentication</h1></body></html>")
}

func main() {
	authenticator := auth.NewBasicAuthenticator("example.com", Secret)
	http.HandleFunc("/", auth.JustCheck(authenticator, regular_handler))
	http.ListenAndServe(":8080", nil)
}
