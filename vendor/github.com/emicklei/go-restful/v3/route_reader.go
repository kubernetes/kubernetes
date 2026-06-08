package restful

// Copyright 2021 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

type RouteReader interface {
	Method() string
	Consumes() []string
	Path() string
	Doc() string
	Notes() string
	Operation() string
	ParameterDocs() []*Parameter
	// Returns a copy
	Metadata() map[string]interface{}
	Deprecated() bool
}

type routeAccessor struct {
	route *Route
}

func (r routeAccessor) Method() string {
	return r.route.Method
}
func (r routeAccessor) Consumes() []string {
	return r.route.Consumes[:]
}
func (r routeAccessor) Path() string {
	return r.route.Path
}
func (r routeAccessor) Doc() string {
	return r.route.Doc
}
func (r routeAccessor) Notes() string {
	return r.route.Notes
}
func (r routeAccessor) Operation() string {
	return r.route.Operation
}
func (r routeAccessor) ParameterDocs() []*Parameter {
	return r.route.ParameterDocs[:]
}

// Returns a copy
func (r routeAccessor) Metadata() map[string]interface{} {
	return copyMap(r.route.Metadata)
}
func (r routeAccessor) Deprecated() bool {
	return r.route.Deprecated
}

// https://stackoverflow.com/questions/23057785/how-to-copy-a-map
func copyMap(m map[string]interface{}) map[string]interface{} {
	cp := make(map[string]interface{})
	for k, v := range m {
		vm, ok := v.(map[string]interface{})
		if ok {
			cp[k] = copyMap(vm)
		} else {
			cp[k] = v
		}
	}
	return cp
}
