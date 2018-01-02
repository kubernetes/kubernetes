package trustgraph

import "github.com/docker/libtrust"

// TrustGraph represents a graph of authorization mapping
// public keys to nodes and grants between nodes.
type TrustGraph interface {
	// Verifies that the given public key is allowed to perform
	// the given action on the given node according to the trust
	// graph.
	Verify(libtrust.PublicKey, string, uint16) (bool, error)

	// GetGrants returns an array of all grant chains which are used to
	// allow the requested permission.
	GetGrants(libtrust.PublicKey, string, uint16) ([][]*Grant, error)
}

// Grant represents a transfer of permission from one part of the
// trust graph to another. This is the only way to delegate
// permission between two different sub trees in the graph.
type Grant struct {
	// Subject is the namespace being granted
	Subject string

	// Permissions is a bit map of permissions
	Permission uint16

	// Grantee represents the node being granted
	// a permission scope.  The grantee can be
	// either a namespace item or a key id where namespace
	// items will always start with a '/'.
	Grantee string

	// statement represents the statement used to create
	// this object.
	statement *Statement
}

// Permissions
//  Read node 0x01 (can read node, no sub nodes)
//  Write node 0x02 (can write to node object, cannot create subnodes)
//  Read subtree 0x04 (delegates read to each sub node)
//  Write subtree 0x08 (delegates write to each sub node, included create on the subject)
//
// Permission shortcuts
// ReadItem = 0x01
// WriteItem = 0x03
// ReadAccess = 0x07
// WriteAccess = 0x0F
// Delegate = 0x0F
