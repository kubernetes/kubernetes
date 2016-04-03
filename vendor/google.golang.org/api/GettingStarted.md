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
$ go get google.golang.org/api/urlshortener/v1
```

Now it's ready for use in your code.

## Using

Once you've installed a library, you import it like this:

```go
package main

import (
    "golang.org/x/net/context"
    "golang.org/x/oauth2"
    "golang.org/x/oauth2/google"
    "google.golang.org/api/urlshortener/v1"
)
```

The package name, if you don't override it on your import line, is the name of the
API without the version number.  In the case above, just `urlshortener`.

## Instantiating

Each API has a `New` function taking an `*http.Client` and returning an API-specific `*Service`.

You create the service like:

```go
    svc, err := urlshortener.New(httpClient)
```

## OAuth HTTP Client

The HTTP client you pass in to the service must be one that automatically adds
Google-supported Authorization information to the requests.

The best option is to use "golang.org/x/oauth2", an OAuth2 library for Go.
You can see how to set up and use oauth2 with these APIs by checking out the
[example code](https://github.com/google/google-api-go-client/tree/master/examples).

In summary, you need to create an OAuth config:

```go
    var config = &oauth2.Config{
        ClientID:     "", // from https://console.developers.google.com/project/<your-project-id>/apiui/credential
        ClientSecret: "", // from https://console.developers.google.com/project/<your-project-id>/apiui/credential
        Endpoint:     google.Endpoint,
        Scopes:       []string{urlshortener.UrlshortenerScope},
    }
```

Then you need to get an OAuth Token from the user.  This involves sending the user
to a URL (at Google) to grant access to your application (either a web application
or a desktop application), and then the browser redirects to the website or local
application's webserver with the per-user token in the URL.

Once you have that token,

```go
    httpClient := newOAuthClient(context.Background(), config)
```

Then you're good to pass that client to the API's `New` function.

## Using API Keys

Some APIs require passing API keys from your application.
To do this, you can use
[transport.APIKey](http://godoc.org/google.golang.org/api/googleapi/transport#APIKey):

```go
    ctx = context.WithValue(context.Background(), oauth2.HTTPClient, &http.Client{
        Transport: &transport.APIKey{Key: developerKey},
    })
    httpClient := newOAuthClient(ctx, config)
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
