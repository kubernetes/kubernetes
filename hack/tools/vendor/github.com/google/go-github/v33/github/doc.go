// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package github provides a client for using the GitHub API.

Usage:

	import "github.com/google/go-github/v33/github"	// with go modules enabled (GO111MODULE=on or outside GOPATH)
	import "github.com/google/go-github/github"     // with go modules disabled

Construct a new GitHub client, then use the various services on the client to
access different parts of the GitHub API. For example:

	client := github.NewClient(nil)

	// list all organizations for user "willnorris"
	orgs, _, err := client.Organizations.List(ctx, "willnorris", nil)

Some API methods have optional parameters that can be passed. For example:

	client := github.NewClient(nil)

	// list public repositories for org "github"
	opt := &github.RepositoryListByOrgOptions{Type: "public"}
	repos, _, err := client.Repositories.ListByOrg(ctx, "github", opt)

The services of a client divide the API into logical chunks and correspond to
the structure of the GitHub API documentation at
https://docs.github.com/en/free-pro-team@latest/rest/reference/.

NOTE: Using the https://godoc.org/context package, one can easily
pass cancelation signals and deadlines to various services of the client for
handling a request. In case there is no context available, then context.Background()
can be used as a starting point.

For more sample code snippets, head over to the https://github.com/google/go-github/tree/master/example directory.

Authentication

The go-github library does not directly handle authentication. Instead, when
creating a new client, pass an http.Client that can handle authentication for
you. The easiest and recommended way to do this is using the golang.org/x/oauth2
library, but you can always use any other library that provides an http.Client.
If you have an OAuth2 access token (for example, a personal API token), you can
use it with the oauth2 library using:

	import "golang.org/x/oauth2"

	func main() {
		ctx := context.Background()
		ts := oauth2.StaticTokenSource(
			&oauth2.Token{AccessToken: "... your access token ..."},
		)
		tc := oauth2.NewClient(ctx, ts)

		client := github.NewClient(tc)

		// list all repositories for the authenticated user
		repos, _, err := client.Repositories.List(ctx, "", nil)
	}

Note that when using an authenticated Client, all calls made by the client will
include the specified OAuth token. Therefore, authenticated clients should
almost never be shared between different users.

See the oauth2 docs for complete instructions on using that library.

For API methods that require HTTP Basic Authentication, use the
BasicAuthTransport.

GitHub Apps authentication can be provided by the
https://github.com/bradleyfalzon/ghinstallation package.

	import "github.com/bradleyfalzon/ghinstallation"

	func main() {
		// Wrap the shared transport for use with the integration ID 1 authenticating with installation ID 99.
		itr, err := ghinstallation.NewKeyFromFile(http.DefaultTransport, 1, 99, "2016-10-19.private-key.pem")
		if err != nil {
			// Handle error.
		}

		// Use installation transport with client
		client := github.NewClient(&http.Client{Transport: itr})

		// Use client...
	}

Rate Limiting

GitHub imposes a rate limit on all API clients. Unauthenticated clients are
limited to 60 requests per hour, while authenticated clients can make up to
5,000 requests per hour. The Search API has a custom rate limit. Unauthenticated
clients are limited to 10 requests per minute, while authenticated clients
can make up to 30 requests per minute. To receive the higher rate limit when
making calls that are not issued on behalf of a user,
use UnauthenticatedRateLimitedTransport.

The returned Response.Rate value contains the rate limit information
from the most recent API call. If a recent enough response isn't
available, you can use RateLimits to fetch the most up-to-date rate
limit data for the client.

To detect an API rate limit error, you can check if its type is *github.RateLimitError:

	repos, _, err := client.Repositories.List(ctx, "", nil)
	if _, ok := err.(*github.RateLimitError); ok {
		log.Println("hit rate limit")
	}

Learn more about GitHub rate limiting at
https://docs.github.com/en/free-pro-team@latest/rest/reference/#rate-limiting.

Accepted Status

Some endpoints may return a 202 Accepted status code, meaning that the
information required is not yet ready and was scheduled to be gathered on
the GitHub side. Methods known to behave like this are documented specifying
this behavior.

To detect this condition of error, you can check if its type is
*github.AcceptedError:

	stats, _, err := client.Repositories.ListContributorsStats(ctx, org, repo)
	if _, ok := err.(*github.AcceptedError); ok {
		log.Println("scheduled on GitHub side")
	}

Conditional Requests

The GitHub API has good support for conditional requests which will help
prevent you from burning through your rate limit, as well as help speed up your
application. go-github does not handle conditional requests directly, but is
instead designed to work with a caching http.Transport. We recommend using
https://github.com/gregjones/httpcache for that.

Learn more about GitHub conditional requests at
https://docs.github.com/en/free-pro-team@latest/rest/reference/#conditional-requests.

Creating and Updating Resources

All structs for GitHub resources use pointer values for all non-repeated fields.
This allows distinguishing between unset fields and those set to a zero-value.
Helper functions have been provided to easily create these pointers for string,
bool, and int values. For example:

	// create a new private repository named "foo"
	repo := &github.Repository{
		Name:    github.String("foo"),
		Private: github.Bool(true),
	}
	client.Repositories.Create(ctx, "", repo)

Users who have worked with protocol buffers should find this pattern familiar.

Pagination

All requests for resource collections (repos, pull requests, issues, etc.)
support pagination. Pagination options are described in the
github.ListOptions struct and passed to the list methods directly or as an
embedded type of a more specific list options struct (for example
github.PullRequestListOptions). Pages information is available via the
github.Response struct.

	client := github.NewClient(nil)

	opt := &github.RepositoryListByOrgOptions{
		ListOptions: github.ListOptions{PerPage: 10},
	}
	// get all pages of results
	var allRepos []*github.Repository
	for {
		repos, resp, err := client.Repositories.ListByOrg(ctx, "github", opt)
		if err != nil {
			return err
		}
		allRepos = append(allRepos, repos...)
		if resp.NextPage == 0 {
			break
		}
		opt.Page = resp.NextPage
	}

*/
package github
