// Package extensions provides information and interaction with the different extensions available
// for an OpenStack service.
//
// The purpose of OpenStack API extensions is to:
//
// - Introduce new features in the API without requiring a version change.
// - Introduce vendor-specific niche functionality.
// - Act as a proving ground for experimental functionalities that might be included in a future
//   version of the API.
//
// Extensions usually have tags that prevent conflicts with other extensions that define attributes
// or resources with the same names, and with core resources and attributes.
// Because an extension might not be supported by all plug-ins, its availability varies with deployments
// and the specific plug-in.
package extensions
