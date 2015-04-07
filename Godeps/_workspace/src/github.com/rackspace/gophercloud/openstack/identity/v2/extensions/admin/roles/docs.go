// Package roles provides functionality to interact with and control roles on
// the API.
//
// A role represents a personality that a user can assume when performing a
// specific set of operations. If a role includes a set of rights and
// privileges, a user assuming that role inherits those rights and privileges.
//
// When a token is generated, the list of roles that user can assume is returned
// back to them. Services that are being called by that user determine how they
// interpret the set of roles a user has and to which operations or resources
// each role grants access.
//
// It is up to individual services such as Compute or Image to assign meaning
// to these roles. As far as the Identity service is concerned, a role is an
// arbitrary name assigned by the user.
package roles
