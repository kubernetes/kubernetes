package restful

// Copyright 2021 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

// ExtensionProperties provides storage of vendor extensions for entities
type ExtensionProperties struct {
	// Extensions vendor extensions used to describe extra functionality
	// (https://swagger.io/docs/specification/2-0/swagger-extensions/)
	Extensions map[string]interface{}
}

// AddExtension adds or updates a key=value pair to the extension map.
func (ep *ExtensionProperties) AddExtension(key string, value interface{}) {
	if ep.Extensions == nil {
		ep.Extensions = map[string]interface{}{key: value}
	} else {
		ep.Extensions[key] = value
	}
}
