HTTP Authentication implementation in Go
========================================

This is an implementation of HTTP Basic and HTTP Digest authentication
in Go language. It is designed as a simple wrapper for
http.RequestHandler functions.

Features
--------
 
 * Supports HTTP Basic and HTTP Digest authentication.
 * Supports htpasswd and htdigest formatted files.
 * Automatic reloading of password files.
 * Pluggable interface for user/password storage.
 * Supports MD5 and SHA1 for Basic authentication password storage.
 * Configurable Digest nonce cache size with expiration.
 * Wrapper for legacy http handlers (http.HandlerFunc interface)
 
Example usage
-------------

This is a complete working example for Basic auth:

    package main

    import (
            auth "github.com/abbot/go-http-auth"
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

    func handle(w http.ResponseWriter, r *auth.AuthenticatedRequest) {
            fmt.Fprintf(w, "<html><body><h1>Hello, %s!</h1></body></html>", r.Username)
    }

    func main() {
            authenticator := auth.NewBasicAuthenticator("example.com", Secret)
            http.HandleFunc("/", authenticator.Wrap(handle))
            http.ListenAndServe(":8080", nil)
    }

See more examples in the "examples" directory.

Legal
-----

This module is developed under Apache 2.0 license, and can be used for
open and proprietary projects.

Copyright 2012-2013 Lev Shamardin

Licensed under the Apache License, Version 2.0 (the "License"); you
may not use this file or any other part of this project except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
