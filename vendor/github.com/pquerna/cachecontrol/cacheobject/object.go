/**
 *  Copyright 2015 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package cacheobject

import (
	"net/http"
	"time"
)

// LOW LEVEL API: Repersents a potentially cachable HTTP object.
//
// This struct is designed to be serialized efficiently, so in a high
// performance caching server, things like Date-Strings don't need to be
// parsed for every use of a cached object.
type Object struct {
	CacheIsPrivate bool

	RespDirectives         *ResponseCacheDirectives
	RespHeaders            http.Header
	RespStatusCode         int
	RespExpiresHeader      time.Time
	RespDateHeader         time.Time
	RespLastModifiedHeader time.Time

	ReqDirectives *RequestCacheDirectives
	ReqHeaders    http.Header
	ReqMethod     string

	NowUTC time.Time
}

// LOW LEVEL API: Repersents the results of examinig an Object with
// CachableObject and ExpirationObject.
//
// TODO(pquerna): decide if this is a good idea or bad
type ObjectResults struct {
	OutReasons        []Reason
	OutWarnings       []Warning
	OutExpirationTime time.Time
	OutErr            error
}

// LOW LEVEL API: Check if a object is cachable.
func CachableObject(obj *Object, rv *ObjectResults) {
	rv.OutReasons = nil
	rv.OutWarnings = nil
	rv.OutErr = nil

	switch obj.ReqMethod {
	case "GET":
		break
	case "HEAD":
		break
	case "POST":
		/**
		  POST: http://tools.ietf.org/html/rfc7231#section-4.3.3

		  Responses to POST requests are only cacheable when they include
		  explicit freshness information (see Section 4.2.1 of [RFC7234]).
		  However, POST caching is not widely implemented.  For cases where an
		  origin server wishes the client to be able to cache the result of a
		  POST in a way that can be reused by a later GET, the origin server
		  MAY send a 200 (OK) response containing the result and a
		  Content-Location header field that has the same value as the POST's
		  effective request URI (Section 3.1.4.2).
		*/
		if !hasFreshness(obj.ReqDirectives, obj.RespDirectives, obj.RespHeaders, obj.RespExpiresHeader, obj.CacheIsPrivate) {
			rv.OutReasons = append(rv.OutReasons, ReasonRequestMethodPOST)
		}

	case "PUT":
		rv.OutReasons = append(rv.OutReasons, ReasonRequestMethodPUT)

	case "DELETE":
		rv.OutReasons = append(rv.OutReasons, ReasonRequestMethodDELETE)

	case "CONNECT":
		rv.OutReasons = append(rv.OutReasons, ReasonRequestMethodCONNECT)

	case "OPTIONS":
		rv.OutReasons = append(rv.OutReasons, ReasonRequestMethodOPTIONS)

	case "TRACE":
		rv.OutReasons = append(rv.OutReasons, ReasonRequestMethodTRACE)

	// HTTP Extension Methods: http://www.iana.org/assignments/http-methods/http-methods.xhtml
	//
	// To my knowledge, none of them are cachable. Please open a ticket if this is not the case!
	//
	default:
		rv.OutReasons = append(rv.OutReasons, ReasonRequestMethodUnkown)
	}

	if obj.ReqDirectives.NoStore {
		rv.OutReasons = append(rv.OutReasons, ReasonRequestNoStore)
	}

	// Storing Responses to Authenticated Requests: http://tools.ietf.org/html/rfc7234#section-3.2
	authz := obj.ReqHeaders.Get("Authorization")
	if authz != "" {
		if obj.RespDirectives.MustRevalidate ||
			obj.RespDirectives.Public ||
			obj.RespDirectives.SMaxAge != -1 {
			// Expires of some kind present, this is potentially OK.
		} else {
			rv.OutReasons = append(rv.OutReasons, ReasonRequestAuthorizationHeader)
		}
	}

	if obj.RespDirectives.PrivatePresent && !obj.CacheIsPrivate {
		rv.OutReasons = append(rv.OutReasons, ReasonResponsePrivate)
	}

	if obj.RespDirectives.NoStore {
		rv.OutReasons = append(rv.OutReasons, ReasonResponseNoStore)
	}

	/*
	   the response either:

	     *  contains an Expires header field (see Section 5.3), or

	     *  contains a max-age response directive (see Section 5.2.2.8), or

	     *  contains a s-maxage response directive (see Section 5.2.2.9)
	        and the cache is shared, or

	     *  contains a Cache Control Extension (see Section 5.2.3) that
	        allows it to be cached, or

	     *  has a status code that is defined as cacheable by default (see
	        Section 4.2.2), or

	     *  contains a public response directive (see Section 5.2.2.5).
	*/

	expires := obj.RespHeaders.Get("Expires") != ""
	statusCachable := cachableStatusCode(obj.RespStatusCode)

	if expires ||
		obj.RespDirectives.MaxAge != -1 ||
		(obj.RespDirectives.SMaxAge != -1 && !obj.CacheIsPrivate) ||
		statusCachable ||
		obj.RespDirectives.Public {
		/* cachable by default, at least one of the above conditions was true */
	} else {
		rv.OutReasons = append(rv.OutReasons, ReasonResponseUncachableByDefault)
	}
}

var twentyFourHours = time.Duration(24 * time.Hour)

const debug = false

// LOW LEVEL API: Update an objects expiration time.
func ExpirationObject(obj *Object, rv *ObjectResults) {
	/**
	 * Okay, lets calculate Freshness/Expiration now. woo:
	 *  http://tools.ietf.org/html/rfc7234#section-4.2
	 */

	/*
	   o  If the cache is shared and the s-maxage response directive
	      (Section 5.2.2.9) is present, use its value, or

	   o  If the max-age response directive (Section 5.2.2.8) is present,
	      use its value, or

	   o  If the Expires response header field (Section 5.3) is present, use
	      its value minus the value of the Date response header field, or

	   o  Otherwise, no explicit expiration time is present in the response.
	      A heuristic freshness lifetime might be applicable; see
	      Section 4.2.2.
	*/

	var expiresTime time.Time

	if obj.RespDirectives.SMaxAge != -1 && !obj.CacheIsPrivate {
		expiresTime = obj.NowUTC.Add(time.Second * time.Duration(obj.RespDirectives.SMaxAge))
	} else if obj.RespDirectives.MaxAge != -1 {
		expiresTime = obj.NowUTC.UTC().Add(time.Second * time.Duration(obj.RespDirectives.MaxAge))
	} else if !obj.RespExpiresHeader.IsZero() {
		serverDate := obj.RespDateHeader
		if serverDate.IsZero() {
			// common enough case when a Date: header has not yet been added to an
			// active response.
			serverDate = obj.NowUTC
		}
		expiresTime = obj.NowUTC.Add(obj.RespExpiresHeader.Sub(serverDate))
	} else if !obj.RespLastModifiedHeader.IsZero() {
		// heuristic freshness lifetime
		rv.OutWarnings = append(rv.OutWarnings, WarningHeuristicExpiration)

		// http://httpd.apache.org/docs/2.4/mod/mod_cache.html#cachelastmodifiedfactor
		// CacheMaxExpire defaults to 24 hours
		// CacheLastModifiedFactor: is 0.1
		//
		// expiry-period = MIN(time-since-last-modified-date * factor, 24 hours)
		//
		// obj.NowUTC

		since := obj.RespLastModifiedHeader.Sub(obj.NowUTC)
		since = time.Duration(float64(since) * -0.1)

		if since > twentyFourHours {
			expiresTime = obj.NowUTC.Add(twentyFourHours)
		} else {
			expiresTime = obj.NowUTC.Add(since)
		}

		if debug {
			println("Now UTC: ", obj.NowUTC.String())
			println("Last-Modified: ", obj.RespLastModifiedHeader.String())
			println("Since: ", since.String())
			println("TwentyFourHours: ", twentyFourHours.String())
			println("Expiration: ", expiresTime.String())
		}
	} else {
		// TODO(pquerna): what should the default behavoir be for expiration time?
	}

	rv.OutExpirationTime = expiresTime
}

// Evaluate cachability based on an HTTP request, and parts of the response.
func UsingRequestResponse(req *http.Request,
	statusCode int,
	respHeaders http.Header,
	privateCache bool) ([]Reason, time.Time, error) {

	var reqHeaders http.Header
	var reqMethod string

	var reqDir *RequestCacheDirectives = nil
	respDir, err := ParseResponseCacheControl(respHeaders.Get("Cache-Control"))
	if err != nil {
		return nil, time.Time{}, err
	}

	if req != nil {
		reqDir, err = ParseRequestCacheControl(req.Header.Get("Cache-Control"))
		if err != nil {
			return nil, time.Time{}, err
		}
		reqHeaders = req.Header
		reqMethod = req.Method
	}

	var expiresHeader time.Time
	var dateHeader time.Time
	var lastModifiedHeader time.Time

	if respHeaders.Get("Expires") != "" {
		expiresHeader, err = http.ParseTime(respHeaders.Get("Expires"))
		if err != nil {
			// sometimes servers will return `Expires: 0` or `Expires: -1` to
			// indicate expired content
			expiresHeader = time.Time{}
		}
		expiresHeader = expiresHeader.UTC()
	}

	if respHeaders.Get("Date") != "" {
		dateHeader, err = http.ParseTime(respHeaders.Get("Date"))
		if err != nil {
			return nil, time.Time{}, err
		}
		dateHeader = dateHeader.UTC()
	}

	if respHeaders.Get("Last-Modified") != "" {
		lastModifiedHeader, err = http.ParseTime(respHeaders.Get("Last-Modified"))
		if err != nil {
			return nil, time.Time{}, err
		}
		lastModifiedHeader = lastModifiedHeader.UTC()
	}

	obj := Object{
		CacheIsPrivate: privateCache,

		RespDirectives:         respDir,
		RespHeaders:            respHeaders,
		RespStatusCode:         statusCode,
		RespExpiresHeader:      expiresHeader,
		RespDateHeader:         dateHeader,
		RespLastModifiedHeader: lastModifiedHeader,

		ReqDirectives: reqDir,
		ReqHeaders:    reqHeaders,
		ReqMethod:     reqMethod,

		NowUTC: time.Now().UTC(),
	}
	rv := ObjectResults{}

	CachableObject(&obj, &rv)
	if rv.OutErr != nil {
		return nil, time.Time{}, rv.OutErr
	}

	ExpirationObject(&obj, &rv)
	if rv.OutErr != nil {
		return nil, time.Time{}, rv.OutErr
	}

	return rv.OutReasons, rv.OutExpirationTime, nil
}

// calculate if a freshness directive is present: http://tools.ietf.org/html/rfc7234#section-4.2.1
func hasFreshness(reqDir *RequestCacheDirectives, respDir *ResponseCacheDirectives, respHeaders http.Header, respExpires time.Time, privateCache bool) bool {
	if !privateCache && respDir.SMaxAge != -1 {
		return true
	}

	if respDir.MaxAge != -1 {
		return true
	}

	if !respExpires.IsZero() || respHeaders.Get("Expires") != "" {
		return true
	}

	return false
}

func cachableStatusCode(statusCode int) bool {
	/*
		Responses with status codes that are defined as cacheable by default
		(e.g., 200, 203, 204, 206, 300, 301, 404, 405, 410, 414, and 501 in
		this specification) can be reused by a cache with heuristic
		expiration unless otherwise indicated by the method definition or
		explicit cache controls [RFC7234]; all other status codes are not
		cacheable by default.
	*/
	switch statusCode {
	case 200:
		return true
	case 203:
		return true
	case 204:
		return true
	case 206:
		return true
	case 300:
		return true
	case 301:
		return true
	case 404:
		return true
	case 405:
		return true
	case 410:
		return true
	case 414:
		return true
	case 501:
		return true
	default:
		return false
	}
}
