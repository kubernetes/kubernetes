// Package netlink provides low-level access to Linux netlink sockets
// (AF_NETLINK).
//
// If you have any questions or you'd like some guidance, please join us on
// Gophers Slack (https://invite.slack.golangbridge.org) in the #networking
// channel!
//
// # Network namespaces
//
// This package is aware of Linux network namespaces, and can enter different
// network namespaces either implicitly or explicitly, depending on
// configuration. The Config structure passed to Dial to create a Conn controls
// these behaviors. See the documentation of Config.NetNS for details.
//
// # Debugging
//
// This package supports rudimentary netlink connection debugging support. To
// enable this, run your binary with the NLDEBUG environment variable set.
// Debugging information will be output to stderr with a prefix of "nl:".
//
// To use the debugging defaults, use:
//
//	$ NLDEBUG=1 ./nlctl
//
// To configure individual aspects of the debugger, pass key/value options such
// as:
//
//	$ NLDEBUG=level=1 ./nlctl
//
// Available key/value debugger options include:
//
//	level=N: specify the debugging level (only "1" is currently supported)
package netlink
