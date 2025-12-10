// Package netns allows ultra-simple network namespace handling. NsHandles
// can be retrieved and set. Note that the current namespace is thread
// local so actions that set and reset namespaces should use LockOSThread
// to make sure the namespace doesn't change due to a goroutine switch.
// It is best to close NsHandles when you are done with them. This can be
// accomplished via a `defer ns.Close()` on the handle. Changing namespaces
// requires elevated privileges, so in most cases this code needs to be run
// as root.
package netns
