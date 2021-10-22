// Copyright 2014 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
Package cloud is the root of the packages used to access Google Cloud
Services. See https://godoc.org/cloud.google.com/go for a full list
of sub-packages.


Client Options

All clients in sub-packages are configurable via client options. These options are
described here: https://godoc.org/google.golang.org/api/option.


Authentication and Authorization

All the clients in sub-packages support authentication via Google Application Default
Credentials (see https://cloud.google.com/docs/authentication/production), or
by providing a JSON key file for a Service Account. See the authentication examples
in this package for details.


Timeouts and Cancellation

By default, all requests in sub-packages will run indefinitely, retrying on transient
errors when correctness allows. To set timeouts or arrange for cancellation, use
contexts. See the examples for details.

Do not attempt to control the initial connection (dialing) of a service by setting a
timeout on the context passed to NewClient. Dialing is non-blocking, so timeouts
would be ineffective and would only interfere with credential refreshing, which uses
the same context.


Connection Pooling

Connection pooling differs in clients based on their transport. Cloud
clients either rely on HTTP or gRPC transports to communicate
with Google Cloud.

Cloud clients that use HTTP (bigquery, compute, storage, and translate) rely on the
underlying HTTP transport to cache connections for later re-use. These are cached to
the default http.MaxIdleConns and http.MaxIdleConnsPerHost settings in
http.DefaultTransport.

For gRPC clients (all others in this repo), connection pooling is configurable. Users
of cloud client libraries may specify option.WithGRPCConnectionPool(n) as a client
option to NewClient calls. This configures the underlying gRPC connections to be
pooled and addressed in a round robin fashion.


Using the Libraries with Docker

Minimal docker images like Alpine lack CA certificates. This causes RPCs to appear to
hang, because gRPC retries indefinitely. See https://github.com/googleapis/google-cloud-go/issues/928
for more information.


Debugging

To see gRPC logs, set the environment variable GRPC_GO_LOG_SEVERITY_LEVEL. See
https://godoc.org/google.golang.org/grpc/grpclog for more information.

For HTTP logging, set the GODEBUG environment variable to "http2debug=1" or "http2debug=2".


Client Stability

Clients in this repository are considered alpha or beta unless otherwise
marked as stable in the README.md. Semver is not used to communicate stability
of clients.

Alpha and beta clients may change or go away without notice.

Clients marked stable will maintain compatibility with future versions for as
long as we can reasonably sustain. Incompatible changes might be made in some
situations, including:

- Security bugs may prompt backwards-incompatible changes.

- Situations in which components are no longer feasible to maintain without
making breaking changes, including removal.

- Parts of the client surface may be outright unstable and subject to change.
These parts of the surface will be labeled with the note, "It is EXPERIMENTAL
and subject to change or removal without notice."
*/
package cloud // import "cloud.google.com/go"
