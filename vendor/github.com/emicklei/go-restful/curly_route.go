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

type sortableCurlyRoutes struct {
	candidates []*curlyRoute
}

func (s *sortableCurlyRoutes) add(route *curlyRoute) {
	s.candidates = append(s.candidates, route)
}

func (s *sortableCurlyRoutes) routes() (routes []Route) {
	for _, each := range s.candidates {
		routes = append(routes, each.route) // TODO change return type
	}
	return routes
}

func (s *sortableCurlyRoutes) Len() int {
	return len(s.candidates)
}
func (s *sortableCurlyRoutes) Swap(i, j int) {
	s.candidates[i], s.candidates[j] = s.candidates[j], s.candidates[i]
}
func (s *sortableCurlyRoutes) Less(i, j int) bool {
	ci := s.candidates[i]
	cj := s.candidates[j]

	// primary key
	if ci.staticCount < cj.staticCount {
		return true
	}
	if ci.staticCount > cj.staticCount {
		return false
	}
	// secundary key
	if ci.paramCount < cj.paramCount {
		return true
	}
	if ci.paramCount > cj.paramCount {
		return false
	}
	return ci.route.Path < cj.route.Path
}
