/*
Package goproxy provides a customizable HTTP proxy,
supporting hijacking HTTPS connection.

The intent of the proxy, is to be usable with reasonable amount of traffic
yet, customizable and programable.

The proxy itself is simply an `net/http` handler.

Typical usage is

	proxy := goproxy.NewProxyHttpServer()
	proxy.OnRequest(..conditions..).Do(..requesthandler..)
	proxy.OnRequest(..conditions..).DoFunc(..requesthandlerFunction..)
	proxy.OnResponse(..conditions..).Do(..responesHandler..)
	proxy.OnResponse(..conditions..).DoFunc(..responesHandlerFunction..)
	http.ListenAndServe(":8080", proxy)

Adding a header to each request

	proxy.OnRequest().DoFunc(func(r *http.Request,ctx *goproxy.ProxyCtx) (*http.Request, *http.Response){
		r.Header.Set("X-GoProxy","1")
		return r, nil
	})

Note that the function is called before the proxy sends the request to the server

For printing the content type of all incoming responses

	proxy.OnResponse().DoFunc(func(r *http.Response, ctx *goproxy.ProxyCtx)*http.Response{
		println(ctx.Req.Host,"->",r.Header.Get("Content-Type"))
		return r
	})

note that we used the ProxyCtx context variable here. It contains the request
and the response (Req and Resp, Resp is nil if unavailable) of this specific client
interaction with the proxy.

To print the content type of all responses from a certain url, we'll add a
ReqCondition to the OnResponse function:

	proxy.OnResponse(goproxy.UrlIs("golang.org/pkg")).DoFunc(func(r *http.Response, ctx *goproxy.ProxyCtx)*http.Response{
		println(ctx.Req.Host,"->",r.Header.Get("Content-Type"))
		return r
	})

We can write the condition ourselves, conditions can be set on request and on response

	var random = ReqConditionFunc(func(r *http.Request) bool {
		return rand.Intn(1) == 0
	})
	var hasGoProxyHeader = RespConditionFunc(func(resp *http.Response,req *http.Request)bool {
		return resp.Header.Get("X-GoProxy") != ""
	})

Caution! If you give a RespCondition to the OnRequest function, you'll get a run time panic! It doesn't
make sense to read the response, if you still haven't got it!

Finally, we have convenience function to throw a quick response

	proxy.OnResponse(hasGoProxyHeader).DoFunc(func(r*http.Response,ctx *goproxy.ProxyCtx)*http.Response {
		r.Body.Close()
		return goproxy.NewResponse(ctx.Req, goproxy.ContentTypeText, http.StatusForbidden, "Can't see response with X-GoProxy header!")
	})

we close the body of the original repsonse, and return a new 403 response with a short message.

Example use cases:

1. https://github.com/elazarl/goproxy/tree/master/examples/goproxy-avgsize

To measure the average size of an Html served in your site. One can ask
all the QA team to access the website by a proxy, and the proxy will
measure the average size of all text/html responses from your host.

2. [not yet implemented]

All requests to your web servers should be directed through the proxy,
when the proxy will detect html pieces sent as a response to AJAX
request, it'll send a warning email.

3. https://github.com/elazarl/goproxy/blob/master/examples/goproxy-httpdump/

Generate a real traffic to your website by real users using through
proxy. Record the traffic, and try it again for more real load testing.

4. https://github.com/elazarl/goproxy/tree/master/examples/goproxy-no-reddit-at-worktime

Will allow browsing to reddit.com between 8:00am and 17:00pm

5. https://github.com/elazarl/goproxy/tree/master/examples/goproxy-jquery-version

Will warn if multiple versions of jquery are used in the same domain.

6. https://github.com/elazarl/goproxy/blob/master/examples/goproxy-upside-down-ternet/

Modifies image files in an HTTP response via goproxy's image extension found in ext/.

*/
package goproxy
