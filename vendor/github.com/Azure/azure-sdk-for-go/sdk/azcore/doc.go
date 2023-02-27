//go:build go1.18
// +build go1.18

// Copyright 2017 Microsoft Corporation. All rights reserved.
// Use of this source code is governed by an MIT
// license that can be found in the LICENSE file.

/*
Package azcore implements an HTTP request/response middleware pipeline used by Azure SDK clients.

The middleware consists of three components.

   - One or more Policy instances.
   - A Transporter instance.
   - A Pipeline instance that combines the Policy and Transporter instances.

Implementing the Policy Interface

A Policy can be implemented in two ways; as a first-class function for a stateless Policy, or as
a method on a type for a stateful Policy.  Note that HTTP requests made via the same pipeline share
the same Policy instances, so if a Policy mutates its state it MUST be properly synchronized to
avoid race conditions.

A Policy's Do method is called when an HTTP request wants to be sent over the network. The Do method can
perform any operation(s) it desires. For example, it can log the outgoing request, mutate the URL, headers,
and/or query parameters, inject a failure, etc.  Once the Policy has successfully completed its request
work, it must call the Next() method on the *policy.Request instance in order to pass the request to the
next Policy in the chain.

When an HTTP response comes back, the Policy then gets a chance to process the response/error.  The Policy instance
can log the response, retry the operation if it failed due to a transient error or timeout, unmarshal the response
body, etc.  Once the Policy has successfully completed its response work, it must return the *http.Response
and error instances to its caller.

Template for implementing a stateless Policy:

   type policyFunc func(*policy.Request) (*http.Response, error)
   // Do implements the Policy interface on policyFunc.

   func (pf policyFunc) Do(req *policy.Request) (*http.Response, error) {
	   return pf(req)
   }

   func NewMyStatelessPolicy() policy.Policy {
      return policyFunc(func(req *policy.Request) (*http.Response, error) {
         // TODO: mutate/process Request here

         // forward Request to next Policy & get Response/error
         resp, err := req.Next()

         // TODO: mutate/process Response/error here

         // return Response/error to previous Policy
         return resp, err
      })
   }

Template for implementing a stateful Policy:

   type MyStatefulPolicy struct {
      // TODO: add configuration/setting fields here
   }

   // TODO: add initialization args to NewMyStatefulPolicy()
   func NewMyStatefulPolicy() policy.Policy {
      return &MyStatefulPolicy{
         // TODO: initialize configuration/setting fields here
      }
   }

   func (p *MyStatefulPolicy) Do(req *policy.Request) (resp *http.Response, err error) {
         // TODO: mutate/process Request here

         // forward Request to next Policy & get Response/error
         resp, err := req.Next()

         // TODO: mutate/process Response/error here

         // return Response/error to previous Policy
         return resp, err
   }

Implementing the Transporter Interface

The Transporter interface is responsible for sending the HTTP request and returning the corresponding
HTTP response or error.  The Transporter is invoked by the last Policy in the chain.  The default Transporter
implementation uses a shared http.Client from the standard library.

The same stateful/stateless rules for Policy implementations apply to Transporter implementations.

Using Policy and Transporter Instances Via a Pipeline

To use the Policy and Transporter instances, an application passes them to the runtime.NewPipeline function.

   func NewPipeline(transport Transporter, policies ...Policy) Pipeline

The specified Policy instances form a chain and are invoked in the order provided to NewPipeline
followed by the Transporter.

Once the Pipeline has been created, create a runtime.Request instance and pass it to Pipeline's Do method.

   func NewRequest(ctx context.Context, httpMethod string, endpoint string) (*Request, error)

   func (p Pipeline) Do(req *Request) (*http.Request, error)

The Pipeline.Do method sends the specified Request through the chain of Policy and Transporter
instances.  The response/error is then sent through the same chain of Policy instances in reverse
order.  For example, assuming there are Policy types PolicyA, PolicyB, and PolicyC along with
TransportA.

   pipeline := NewPipeline(TransportA, PolicyA, PolicyB, PolicyC)

The flow of Request and Response looks like the following:

   policy.Request -> PolicyA -> PolicyB -> PolicyC -> TransportA -----+
                                                                      |
                                                               HTTP(S) endpoint
                                                                      |
   caller <--------- PolicyA <- PolicyB <- PolicyC <- http.Response-+

Creating a Request Instance

The Request instance passed to Pipeline's Do method is a wrapper around an *http.Request.  It also
contains some internal state and provides various convenience methods.  You create a Request instance
by calling the runtime.NewRequest function:

   func NewRequest(ctx context.Context, httpMethod string, endpoint string) (*Request, error)

If the Request should contain a body, call the SetBody method.

   func (req *Request) SetBody(body ReadSeekCloser, contentType string) error

A seekable stream is required so that upon retry, the retry Policy instance can seek the stream
back to the beginning before retrying the network request and re-uploading the body.

Sending an Explicit Null

Operations like JSON-MERGE-PATCH send a JSON null to indicate a value should be deleted.

   {
      "delete-me": null
   }

This requirement conflicts with the SDK's default marshalling that specifies "omitempty" as
a means to resolve the ambiguity between a field to be excluded and its zero-value.

   type Widget struct {
      Name  *string `json:",omitempty"`
      Count *int    `json:",omitempty"`
   }

In the above example, Name and Count are defined as pointer-to-type to disambiguate between
a missing value (nil) and a zero-value (0) which might have semantic differences.

In a PATCH operation, any fields left as nil are to have their values preserved.  When updating
a Widget's count, one simply specifies the new value for Count, leaving Name nil.

To fulfill the requirement for sending a JSON null, the NullValue() function can be used.

   w := Widget{
      Count: azcore.NullValue[*int](),
   }

This sends an explict "null" for Count, indicating that any current value for Count should be deleted.

Processing the Response

When the HTTP response is received, the *http.Response is returned directly. Each Policy instance
can inspect/mutate the *http.Response.

Built-in Logging

To enable logging, set environment variable AZURE_SDK_GO_LOGGING to "all" before executing your program.

By default the logger writes to stderr.  This can be customized by calling log.SetListener, providing
a callback that writes to the desired location.  Any custom logging implementation MUST provide its
own synchronization to handle concurrent invocations.

See the docs for the log package for further details.

Pageable Operations

Pageable operations return potentially large data sets spread over multiple GET requests.  The result of
each GET is a "page" of data consisting of a slice of items.

Pageable operations can be identified by their New*Pager naming convention and return type of *runtime.Pager[T].

   func (c *WidgetClient) NewListWidgetsPager(o *Options) *runtime.Pager[PageResponse]

The call to WidgetClient.NewListWidgetsPager() returns an instance of *runtime.Pager[T] for fetching pages
and determining if there are more pages to fetch.  No IO calls are made until the NextPage() method is invoked.

   pager := widgetClient.NewListWidgetsPager(nil)
   for pager.More() {
      page, err := pager.NextPage(context.TODO())
      // handle err
      for _, widget := range page.Values {
         // process widget
      }
   }

Long-Running Operations

Long-running operations (LROs) are operations consisting of an initial request to start the operation followed
by polling to determine when the operation has reached a terminal state.  An LRO's terminal state is one
of the following values.

   * Succeeded - the LRO completed successfully
   *    Failed - the LRO failed to complete
   *  Canceled - the LRO was canceled

LROs can be identified by their Begin* prefix and their return type of *runtime.Poller[T].

   func (c *WidgetClient) BeginCreateOrUpdate(ctx context.Context, w Widget, o *Options) (*runtime.Poller[Response], error)

When a call to WidgetClient.BeginCreateOrUpdate() returns a nil error, it means that the LRO has started.
It does _not_ mean that the widget has been created or updated (or failed to be created/updated).

The *runtime.Poller[T] provides APIs for determining the state of the LRO.  To wait for the LRO to complete,
call the PollUntilDone() method.

   poller, err := widgetClient.BeginCreateOrUpdate(context.TODO(), Widget{}, nil)
   // handle err
   result, err := poller.PollUntilDone(context.TODO(), nil)
   // handle err
   // use result

The call to PollUntilDone() will block the current goroutine until the LRO has reached a terminal state or the
context is canceled/timed out.

Note that LROs can take anywhere from several seconds to several minutes.  The duration is operation-dependent.  Due to
this variant behavior, pollers do _not_ have a preconfigured time-out.  Use a context with the appropriate cancellation
mechanism as required.

Resume Tokens

Pollers provide the ability to serialize their state into a "resume token" which can be used by another process to
recreate the poller.  This is achieved via the runtime.Poller[T].ResumeToken() method.

   token, err := poller.ResumeToken()
   // handle error

Note that a token can only be obtained for a poller that's in a non-terminal state.  Also note that any subsequent calls
to poller.Poll() might change the poller's state.  In this case, a new token should be created.

After the token has been obtained, it can be used to recreate an instance of the originating poller.

   poller, err := widgetClient.BeginCreateOrUpdate(nil, Widget{}, &Options{
      ResumeToken: token,
   })

When resuming a poller, no IO is performed, and zero-value arguments can be used for everything but the Options.ResumeToken.

Resume tokens are unique per service client and operation.  Attempting to resume a poller for LRO BeginB() with a token from LRO
BeginA() will result in an error.
*/
package azcore
