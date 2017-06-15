
## On Pull Requests

- Before you start a PR there needs to be a Github issue and a discussion about it
  on that issue with a core contributor, even if it's just a 'SGTM'.

- A PR's description must reference the issue it closes with a `For <ISSUE NUMBER>` (e.g. For #293).

- A PR's description must contain link(s) to the line(s) in the OpenStack
  source code (on Github) that prove(s) the PR code to be valid. Links to documentation
  are not good enough. The link(s) should be to a non-`master` branch. For example,
  a pull request implementing the creation of a Neutron v2 subnet might put the
  following link in the description:

  https://github.com/openstack/neutron/blob/stable/mitaka/neutron/api/v2/attributes.py#L749

  From that link, a reviewer (or user) can verify the fields in the request/response
  objects in the PR.

- A PR that is in-progress should have `[wip]` in front of the PR's title. When
  ready for review, remove the `[wip]` and ping a core contributor with an `@`.

- Forcing PRs to be small can have the effect of users submitting PRs in a hierarchical chain, with
  one depending on the next. If a PR depends on another one, it should have a [Pending #PRNUM]
  prefix in the PR title. In addition, it will be the PR submitter's responsibility to remove the
  [Pending #PRNUM] tag once the PR has been updated with the merged, dependent PR. That will
  let reviewers know it is ready to review.

- A PR should be small. Even if you intend on implementing an entire
  service, a PR should only be one route of that service
  (e.g. create server or get server, but not both).

- Unless explicitly asked, do not squash commits in the middle of a review; only
  append. It makes it difficult for the reviewer to see what's changed from one
  review to the next.

## On Code

- In re design: follow as closely as is reasonable the code already in the library.
  Most operations (e.g. create, delete) admit the same design.

- Unit tests and acceptance (integration) tests must be written to cover each PR.
  Tests for operations with several options (e.g. list, create) should include all
  the options in the tests. This will allow users to verify an operation on their
  own infrastructure and see an example of usage.

- If in doubt, ask in-line on the PR.

### File Structure

- The following should be used in most cases:

  - `requests.go`: contains all the functions that make HTTP requests and the
    types associated with the HTTP request (parameters for URL, body, etc)
  - `results.go`: contains all the response objects and their methods
  - `urls.go`: contains the endpoints to which the requests are made

### Naming

- For methods on a type in `results.go`, the receiver should be named `r` and the
  variable into which it will be unmarshalled `s`.

- Functions in `requests.go`, with the exception of functions that return a
  `pagination.Pager`, should be named returns of the name `r`.

- Functions in `requests.go` that accept request bodies should accept as their
  last parameter an `interface` named `<Action>OptsBuilder` (eg `CreateOptsBuilder`).
  This `interface` should have at the least a method named `To<Resource><Action>Map`
  (eg `ToPortCreateMap`).

- Functions in `requests.go` that accept query strings should accept as their
  last parameter an `interface` named `<Action>OptsBuilder` (eg `ListOptsBuilder`).
  This `interface` should have at the least a method named `To<Resource><Action>Query`
  (eg `ToServerListQuery`).
