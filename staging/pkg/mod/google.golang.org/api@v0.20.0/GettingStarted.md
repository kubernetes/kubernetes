# Getting Started with the Google APIs for Go

## Getting Started

This is a quick walk-through of how to get started with the Google APIs for Go.

## Background

The first thing to understand is that the Google API libraries are auto-generated for
each language, including Go, so they may not feel like 100% natural for any language.
The Go versions are pretty natural, but please forgive any small non-idiomatic things.
(Suggestions welcome, though!)

## Installing

Pick an API and a version of that API to install.
You can find the complete list by looking at the
[directories here](https://github.com/google/google-api-go-client/tree/master/).

For example, let's install the
[urlshortener's version 1 API](https://godoc.org/google.golang.org/api/urlshortener/v1):

```
$ go get -u google.golang.org/api/urlshortener/v1
```

Now it's ready for use in your code.

## Using

Once you've installed a library, you import it like this:

```go
package main

import (
    "context"
    "golang.org/x/oauth2"
    "golang.org/x/oauth2/google"
    "google.golang.org/api/urlshortener/v1"
)
```

The package name, if you don't override it on your import line, is the name of the
API without the version number. In the case above, just `urlshortener`.

## Instantiating

Each API has a `New` function taking an `*http.Client` and returning an API-specific `*Service`.

You create the service like:

```go
    svc, err := urlshortener.New(httpClient)
```

## OAuth HTTP Client

The HTTP client you pass in to the service must be one that automatically adds
Google-supported Authorization information to the requests.

There are several ways to do authentication. They will all involve the package
[golang.org/x/oauth2](https://godoc.org/golang.org/x/oauth2) in some way.

### 3-legged OAuth

For 3-legged OAuth (your application redirecting a user through a website to get a
token giving your application access to that user's resources), you will need to
create an oauth2.Config,


```go
    var config = &oauth2.Config{
        ClientID:     "", // from https://console.developers.google.com/project/<your-project-id>/apiui/credential
        ClientSecret: "", // from https://console.developers.google.com/project/<your-project-id>/apiui/credential
        Endpoint:     google.Endpoint,
        Scopes:       []string{urlshortener.UrlshortenerScope},
    }
```

... and then use the AuthCodeURL, Exchange, and Client methods on it.
For an example, see: https://godoc.org/golang.org/x/oauth2#example-Config

For the redirect URL, see
https://developers.google.com/identity/protocols/OAuth2InstalledApp#choosingredirecturi

### Service Accounts

To use a Google service account, or the GCE metadata service, see
the [golang.org/x/oauth2/google](https://godoc.org/golang.org/x/oauth2/google) package.
In particular, see [google.DefaultClient](https://godoc.org/golang.org/x/oauth2/google#DefaultClient).

### Using API Keys

Some APIs require passing API keys from your application.
To do this, you can use
[transport.APIKey](https://godoc.org/google.golang.org/api/googleapi/transport#APIKey):

```go
    ctx := context.WithValue(context.Background(), oauth2.HTTPClient, &http.Client{
        Transport: &transport.APIKey{Key: developerKey},
    })
    oauthConfig := &oauth2.Config{ .... }
    var token *oauth2.Token = .... // via cache, or oauthConfig.Exchange
    httpClient := oauthConfig.Client(ctx, token)
    svc, err := urlshortener.New(httpClient)
    ...
```

## Using the Service

Each service contains zero or more methods and zero or more sub-services.
The sub-services related to a specific type of "Resource".

Those sub-services then contain their own methods.

For instance, the urlshortener API has just the "Url" sub-service:

```go
    url, err := svc.Url.Get(shortURL).Do()
    if err != nil {
        ...
    }
    fmt.Printf("The URL %s goes to %s\n", shortURL, url.LongUrl)
```

For a more complete example, see
[urlshortener.go](https://github.com/google/google-api-go-client/tree/master/examples/urlshortener.go)
in the [examples directory](https://github.com/google/google-api-go-client/tree/master/examples/).
(the examples use some functions in `main.go` in the same directory)

## Error Handling

Most errors returned by the `Do` methods of these clients will be of type
[`googleapi.Error`](https://godoc.org/google.golang.org/api/googleapi#Error).
Use a type assertion to obtain the HTTP status code and other properties of the
error:

```go
    url, err := svc.Url.Get(shortURL).Do()
    if err != nil {
        if e, ok := err.(*googleapi.Error); ok && e.Code == http.StatusNotFound {
            ...
        }
    }
```
