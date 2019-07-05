package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

// curlyRoute exits for sorting Routes by the CurlyRouter based on number of parameters and number of static path elements.
type curlyRoute struct {
	route       Route
	paramCount  int
	staticCount int
}

// sortableCurlyRoutes orders by most parameters and path elements first.
type sortableCurlyRoutes []curlyRoute

func (s *sortableCurlyRoutes) add(route curlyRoute) {
	*s = append(*s, route)
}

func (s sortableCurlyRoutes) routes() (routes []Route) {
	routes = make([]Route, 0, len(s))
	for _, each := range s {
		routes = append(routes, each.route) // TODO change return type
	}
	return routes
}

func (s sortableCurlyRoutes) Len() int {
	return len(s)
}
func (s sortableCurlyRoutes) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s sortableCurlyRoutes) Less(i, j int) bool {
	a := s[j]
	b := s[i]

	// primary key
	if a.staticCount < b.staticCount {
		return true
	}
	if a.staticCount > b.staticCount {
		return false
	}
	// secundary key
	if a.paramCount < b.paramCount {
		return true
	}
	if a.paramCount > b.paramCount {
		return false
	}
	return a.route.Path < b.route.Path
}
