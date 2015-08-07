// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mux

import (
	"bytes"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

// newRouteRegexp parses a route template and returns a routeRegexp,
// used to match a host, a path or a query string.
//
// It will extract named variables, assemble a regexp to be matched, create
// a "reverse" template to build URLs and compile regexps to validate variable
// values used in URL building.
//
// Previously we accepted only Python-like identifiers for variable
// names ([a-zA-Z_][a-zA-Z0-9_]*), but currently the only restriction is that
// name and pattern can't be empty, and names can't contain a colon.
func newRouteRegexp(tpl string, matchHost, matchPrefix, matchQuery, strictSlash bool) (*routeRegexp, error) {
	// Check if it is well-formed.
	idxs, errBraces := braceIndices(tpl)
	if errBraces != nil {
		return nil, errBraces
	}
	// Backup the original.
	template := tpl
	// Now let's parse it.
	defaultPattern := "[^/]+"
	if matchQuery {
		defaultPattern = "[^?&]*"
	} else if matchHost {
		defaultPattern = "[^.]+"
		matchPrefix = false
	}
	// Only match strict slash if not matching
	if matchPrefix || matchHost || matchQuery {
		strictSlash = false
	}
	// Set a flag for strictSlash.
	endSlash := false
	if strictSlash && strings.HasSuffix(tpl, "/") {
		tpl = tpl[:len(tpl)-1]
		endSlash = true
	}
	varsN := make([]string, len(idxs)/2)
	varsR := make([]*regexp.Regexp, len(idxs)/2)
	pattern := bytes.NewBufferString("")
	pattern.WriteByte('^')
	reverse := bytes.NewBufferString("")
	var end int
	var err error
	for i := 0; i < len(idxs); i += 2 {
		// Set all values we are interested in.
		raw := tpl[end:idxs[i]]
		end = idxs[i+1]
		parts := strings.SplitN(tpl[idxs[i]+1:end-1], ":", 2)
		name := parts[0]
		patt := defaultPattern
		if len(parts) == 2 {
			patt = parts[1]
		}
		// Name or pattern can't be empty.
		if name == "" || patt == "" {
			return nil, fmt.Errorf("mux: missing name or pattern in %q",
				tpl[idxs[i]:end])
		}
		// Build the regexp pattern.
		fmt.Fprintf(pattern, "%s(%s)", regexp.QuoteMeta(raw), patt)
		// Build the reverse template.
		fmt.Fprintf(reverse, "%s%%s", raw)

		// Append variable name and compiled pattern.
		varsN[i/2] = name
		varsR[i/2], err = regexp.Compile(fmt.Sprintf("^%s$", patt))
		if err != nil {
			return nil, err
		}
	}
	// Add the remaining.
	raw := tpl[end:]
	pattern.WriteString(regexp.QuoteMeta(raw))
	if strictSlash {
		pattern.WriteString("[/]?")
	}
	if matchQuery {
		// Add the default pattern if the query value is empty
		if queryVal := strings.SplitN(template, "=", 2)[1]; queryVal == "" {
			pattern.WriteString(defaultPattern)
		}
	}
	if !matchPrefix {
		pattern.WriteByte('$')
	}
	reverse.WriteString(raw)
	if endSlash {
		reverse.WriteByte('/')
	}
	// Compile full regexp.
	reg, errCompile := regexp.Compile(pattern.String())
	if errCompile != nil {
		return nil, errCompile
	}
	// Done!
	return &routeRegexp{
		template:    template,
		matchHost:   matchHost,
		matchQuery:  matchQuery,
		strictSlash: strictSlash,
		regexp:      reg,
		reverse:     reverse.String(),
		varsN:       varsN,
		varsR:       varsR,
	}, nil
}

// routeRegexp stores a regexp to match a host or path and information to
// collect and validate route variables.
type routeRegexp struct {
	// The unmodified template.
	template string
	// True for host match, false for path or query string match.
	matchHost bool
	// True for query string match, false for path and host match.
	matchQuery bool
	// The strictSlash value defined on the route, but disabled if PathPrefix was used.
	strictSlash bool
	// Expanded regexp.
	regexp *regexp.Regexp
	// Reverse template.
	reverse string
	// Variable names.
	varsN []string
	// Variable regexps (validators).
	varsR []*regexp.Regexp
}

// Match matches the regexp against the URL host or path.
func (r *routeRegexp) Match(req *http.Request, match *RouteMatch) bool {
	if !r.matchHost {
		if r.matchQuery {
			return r.matchQueryString(req)
		} else {
			return r.regexp.MatchString(req.URL.Path)
		}
	}
	return r.regexp.MatchString(getHost(req))
}

// url builds a URL part using the given values.
func (r *routeRegexp) url(values map[string]string) (string, error) {
	urlValues := make([]interface{}, len(r.varsN))
	for k, v := range r.varsN {
		value, ok := values[v]
		if !ok {
			return "", fmt.Errorf("mux: missing route variable %q", v)
		}
		urlValues[k] = value
	}
	rv := fmt.Sprintf(r.reverse, urlValues...)
	if !r.regexp.MatchString(rv) {
		// The URL is checked against the full regexp, instead of checking
		// individual variables. This is faster but to provide a good error
		// message, we check individual regexps if the URL doesn't match.
		for k, v := range r.varsN {
			if !r.varsR[k].MatchString(values[v]) {
				return "", fmt.Errorf(
					"mux: variable %q doesn't match, expected %q", values[v],
					r.varsR[k].String())
			}
		}
	}
	return rv, nil
}

// getUrlQuery returns a single query parameter from a request URL.
// For a URL with foo=bar&baz=ding, we return only the relevant key
// value pair for the routeRegexp.
func (r *routeRegexp) getUrlQuery(req *http.Request) string {
	if !r.matchQuery {
		return ""
	}
	templateKey := strings.SplitN(r.template, "=", 2)[0]
	for key, vals := range req.URL.Query() {
		if key == templateKey && len(vals) > 0 {
			return key + "=" + vals[0]
		}
	}
	return ""
}

func (r *routeRegexp) matchQueryString(req *http.Request) bool {
	return r.regexp.MatchString(r.getUrlQuery(req))
}

// braceIndices returns the first level curly brace indices from a string.
// It returns an error in case of unbalanced braces.
func braceIndices(s string) ([]int, error) {
	var level, idx int
	idxs := make([]int, 0)
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '{':
			if level++; level == 1 {
				idx = i
			}
		case '}':
			if level--; level == 0 {
				idxs = append(idxs, idx, i+1)
			} else if level < 0 {
				return nil, fmt.Errorf("mux: unbalanced braces in %q", s)
			}
		}
	}
	if level != 0 {
		return nil, fmt.Errorf("mux: unbalanced braces in %q", s)
	}
	return idxs, nil
}

// ----------------------------------------------------------------------------
// routeRegexpGroup
// ----------------------------------------------------------------------------

// routeRegexpGroup groups the route matchers that carry variables.
type routeRegexpGroup struct {
	host    *routeRegexp
	path    *routeRegexp
	queries []*routeRegexp
}

// setMatch extracts the variables from the URL once a route matches.
func (v *routeRegexpGroup) setMatch(req *http.Request, m *RouteMatch, r *Route) {
	// Store host variables.
	if v.host != nil {
		hostVars := v.host.regexp.FindStringSubmatch(getHost(req))
		if hostVars != nil {
			for k, v := range v.host.varsN {
				m.Vars[v] = hostVars[k+1]
			}
		}
	}
	// Store path variables.
	if v.path != nil {
		pathVars := v.path.regexp.FindStringSubmatch(req.URL.Path)
		if pathVars != nil {
			for k, v := range v.path.varsN {
				m.Vars[v] = pathVars[k+1]
			}
			// Check if we should redirect.
			if v.path.strictSlash {
				p1 := strings.HasSuffix(req.URL.Path, "/")
				p2 := strings.HasSuffix(v.path.template, "/")
				if p1 != p2 {
					u, _ := url.Parse(req.URL.String())
					if p1 {
						u.Path = u.Path[:len(u.Path)-1]
					} else {
						u.Path += "/"
					}
					m.Handler = http.RedirectHandler(u.String(), 301)
				}
			}
		}
	}
	// Store query string variables.
	for _, q := range v.queries {
		queryVars := q.regexp.FindStringSubmatch(q.getUrlQuery(req))
		if queryVars != nil {
			for k, v := range q.varsN {
				m.Vars[v] = queryVars[k+1]
			}
		}
	}
}

// getHost tries its best to return the request host.
func getHost(r *http.Request) string {
	if r.URL.IsAbs() {
		return r.URL.Host
	}
	host := r.Host
	// Slice off any port information.
	if i := strings.Index(host, ":"); i != -1 {
		host = host[:i]
	}
	return host

}
