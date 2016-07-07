package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"net/http"
	"regexp"
	"sort"
	"strings"
)

// CurlyRouter expects Routes with paths that contain zero or more parameters in curly brackets.
type CurlyRouter struct{}

// SelectRoute is part of the Router interface and returns the best match
// for the WebService and its Route for the given Request.
func (c CurlyRouter) SelectRoute(
	webServices []*WebService,
	httpRequest *http.Request) (selectedService *WebService, selected *Route, err error) {

	requestTokens := tokenizePath(httpRequest.URL.Path)

	detectedService := c.detectWebService(requestTokens, webServices)
	if detectedService == nil {
		if trace {
			traceLogger.Printf("no WebService was found to match URL path:%s\n", httpRequest.URL.Path)
		}
		return nil, nil, NewError(http.StatusNotFound, "404: Page Not Found")
	}
	candidateRoutes := c.selectRoutes(detectedService, requestTokens)
	if len(candidateRoutes) == 0 {
		if trace {
			traceLogger.Printf("no Route in WebService with path %s was found to match URL path:%s\n", detectedService.rootPath, httpRequest.URL.Path)
		}
		return detectedService, nil, NewError(http.StatusNotFound, "404: Page Not Found")
	}
	selectedRoute, err := c.detectRoute(candidateRoutes, httpRequest)
	if selectedRoute == nil {
		return detectedService, nil, err
	}
	return detectedService, selectedRoute, nil
}

// selectRoutes return a collection of Route from a WebService that matches the path tokens from the request.
func (c CurlyRouter) selectRoutes(ws *WebService, requestTokens []string) sortableCurlyRoutes {
	candidates := sortableCurlyRoutes{}
	for _, each := range ws.routes {
		matches, paramCount, staticCount := c.matchesRouteByPathTokens(each.pathParts, requestTokens)
		if matches {
			candidates.add(curlyRoute{each, paramCount, staticCount}) // TODO make sure Routes() return pointers?
		}
	}
	sort.Sort(sort.Reverse(candidates))
	return candidates
}

// matchesRouteByPathTokens computes whether it matches, howmany parameters do match and what the number of static path elements are.
func (c CurlyRouter) matchesRouteByPathTokens(routeTokens, requestTokens []string) (matches bool, paramCount int, staticCount int) {
	if len(routeTokens) < len(requestTokens) {
		// proceed in matching only if last routeToken is wildcard
		count := len(routeTokens)
		if count == 0 || !strings.HasSuffix(routeTokens[count-1], "*}") {
			return false, 0, 0
		}
		// proceed
	}
	for i, routeToken := range routeTokens {
		if i == len(requestTokens) {
			// reached end of request path
			return false, 0, 0
		}
		requestToken := requestTokens[i]
		if strings.HasPrefix(routeToken, "{") {
			paramCount++
			if colon := strings.Index(routeToken, ":"); colon != -1 {
				// match by regex
				matchesToken, matchesRemainder := c.regularMatchesPathToken(routeToken, colon, requestToken)
				if !matchesToken {
					return false, 0, 0
				}
				if matchesRemainder {
					break
				}
			}
		} else { // no { prefix
			if requestToken != routeToken {
				return false, 0, 0
			}
			staticCount++
		}
	}
	return true, paramCount, staticCount
}

// regularMatchesPathToken tests whether the regular expression part of routeToken matches the requestToken or all remaining tokens
// format routeToken is {someVar:someExpression}, e.g. {zipcode:[\d][\d][\d][\d][A-Z][A-Z]}
func (c CurlyRouter) regularMatchesPathToken(routeToken string, colon int, requestToken string) (matchesToken bool, matchesRemainder bool) {
	regPart := routeToken[colon+1 : len(routeToken)-1]
	if regPart == "*" {
		if trace {
			traceLogger.Printf("wildcard parameter detected in route token %s that matches %s\n", routeToken, requestToken)
		}
		return true, true
	}
	matched, err := regexp.MatchString(regPart, requestToken)
	return (matched && err == nil), false
}

// detectRoute selectes from a list of Route the first match by inspecting both the Accept and Content-Type
// headers of the Request. See also RouterJSR311 in jsr311.go
func (c CurlyRouter) detectRoute(candidateRoutes sortableCurlyRoutes, httpRequest *http.Request) (*Route, error) {
	// tracing is done inside detectRoute
	return RouterJSR311{}.detectRoute(candidateRoutes.routes(), httpRequest)
}

// detectWebService returns the best matching webService given the list of path tokens.
// see also computeWebserviceScore
func (c CurlyRouter) detectWebService(requestTokens []string, webServices []*WebService) *WebService {
	var best *WebService
	score := -1
	for _, each := range webServices {
		matches, eachScore := c.computeWebserviceScore(requestTokens, each.pathExpr.tokens)
		if matches && (eachScore > score) {
			best = each
			score = eachScore
		}
	}
	return best
}

// computeWebserviceScore returns whether tokens match and
// the weighted score of the longest matching consecutive tokens from the beginning.
func (c CurlyRouter) computeWebserviceScore(requestTokens []string, tokens []string) (bool, int) {
	if len(tokens) > len(requestTokens) {
		return false, 0
	}
	score := 0
	for i := 0; i < len(tokens); i++ {
		each := requestTokens[i]
		other := tokens[i]
		if len(each) == 0 && len(other) == 0 {
			score++
			continue
		}
		if len(other) > 0 && strings.HasPrefix(other, "{") {
			// no empty match
			if len(each) == 0 {
				return false, score
			}
			score += 1
		} else {
			// not a parameter
			if each != other {
				return false, score
			}
			score += (len(tokens) - i) * 10 //fuzzy
		}
	}
	return true, score
}
