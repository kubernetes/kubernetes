/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package watchtotalorder provides a test plan for validating that Kubernetes watch clients observe a total order of
// events. The Kubernetes API contract ensures that users are able to observe every single event matching a selector,
// using the following procedure:
// 1. issue a LIST call to determine the current state of objects, along with the resourceVersion at which the snapshot
//    was resolved from the API server
// 2. issue a WATCH from that resourceVersion, consume events
// 3. if any transient errors occur on the WATCH stream, use the resourceVersion from the most recent observed event to
//    establish a new watch stream
// 4. if re-establishing the watch fails with code 410 Gone, go back to step 1 and accept that some events will have
//    been lost to this client
//
// In order to provide this semantic, every client must observe the exact same total ordering of watch events. Consider
// the case where a client observes some events out of order, with resourceVersions (1, 2, 4, 3). If the client is to
// reconnect to the watch stream from the last observed resourceVersion (3), it will observe event 4 twice. Similarly,
// if a client observes (1, 2, 4), disconnects and reconnects from the last observed resourceVersion (4), the client
// will never observe event 3. Another way to consider this invariant is to say that when a client observes a watch
// event with resourceVersion R, the client has observed every possible event older than R and no events newer than R.
//
// Testing this invariant in end-to-end conformance tests is critical, as a Kubernetes distribution that does not
// provide this functionality will subtly break all clients expecting to see all events in these re-connection cases.
// Such a test must furthermore make the strong statement that all clients observe *all* events, in the same order. The
// test plan in ths package registers a CRD specific to the test in order to be as certain as possible that all the
// events our watch streams consume come form client interaction that the test itself makes. However, as the conformance
// suite may be executed against any Kubernetes cluster concurrently with any other client interaction to that server,
// the test cannot be strict. A controller that mutates all possible objects on the cluster under test would cause
// spurious events.
//
// Therefore, when executed as part of the integration test suite, the test plan validates that every client observes
// an exact set of events, in the same order. When executed as part of the end-to-end suite, however, the test is
// relaxed to validate that every client observes all the events that the test itself generates, in the same order.
package watchtotalorder
