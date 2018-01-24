go-git-http
===========

[![Build Status](https://travis-ci.org/AaronO/go-git-http.svg)](https://travis-ci.org/AaronO/go-git-http)

A Smart Git Http server library in Go (golang)

### Example

```go
package main

import (
    "log"
    "net/http"

    "github.com/AaronO/go-git-http"
)

func main() {
    // Get git handler to serve a directory of repos
    git := githttp.New("/Users/aaron/git")

    // Attach handler to http server
    http.Handle("/", git)

    // Start HTTP server
    err := http.ListenAndServe(":8080", nil)
    if err != nil {
        log.Fatal("ListenAndServe: ", err)
    }
}
```

### Authentication example

```go
package main

import (
    "log"
    "net/http"

    "github.com/AaronO/go-git-http"
    "github.com/AaronO/go-git-http/auth"
)


func main() {
    // Get git handler to serve a directory of repos
    git := githttp.New("/Users/aaron/git")

    // Build an authentication middleware based on a function
    authenticator := auth.Authenticator(func(info auth.AuthInfo) (bool, error) {
        // Disallow Pushes (making git server pull only)
        if info.Push {
            return false, nil
        }

        // Typically this would be a database lookup
        if info.Username == "admin" && info.Password == "password" {
            return true, nil
        }

        return false, nil
    })

    // Attach handler to http server
    // wrap authenticator around git handler
    http.Handle("/", authenticator(git))

    // Start HTTP server
    err := http.ListenAndServe(":8080", nil)
    if err != nil {
        log.Fatal("ListenAndServe: ", err)
    }
}
```

