package restful

import "strings"

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

// OPTIONSFilter is a filter function that inspects the Http Request for the OPTIONS method
// and provides the response with a set of allowed methods for the request URL Path.
// As for any filter, you can also install it for a particular WebService within a Container.
// Note: this filter is not needed when using CrossOriginResourceSharing (for CORS).
func (c *Container) OPTIONSFilter(req *Request, resp *Response, chain *FilterChain) {
	if "OPTIONS" != req.Request.Method {
		chain.ProcessFilter(req, resp)
		return
	}

	archs := req.Request.Header.Get(HEADER_AccessControlRequestHeaders)
	methods := strings.Join(c.computeAllowedMethods(req), ",")
	origin := req.Request.Header.Get(HEADER_Origin)

	resp.AddHeader(HEADER_Allow, methods)
	resp.AddHeader(HEADER_AccessControlAllowOrigin, origin)
	resp.AddHeader(HEADER_AccessControlAllowHeaders, archs)
	resp.AddHeader(HEADER_AccessControlAllowMethods, methods)
}

// OPTIONSFilter is a filter function that inspects the Http Request for the OPTIONS method
// and provides the response with a set of allowed methods for the request URL Path.
// Note: this filter is not needed when using CrossOriginResourceSharing (for CORS).
func OPTIONSFilter() FilterFunction {
	return DefaultContainer.OPTIONSFilter
}
