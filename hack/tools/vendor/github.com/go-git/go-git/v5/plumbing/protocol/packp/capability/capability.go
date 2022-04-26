// Package capability defines the server and client capabilities.
package capability

// Capability describes a server or client capability.
type Capability string

func (n Capability) String() string {
	return string(n)
}

const (
	// MultiACK capability allows the server to return "ACK obj-id continue" as
	// soon as it finds a commit that it can use as a common base, between the
	// client's wants and the client's have set.
	//
	// By sending this early, the server can potentially head off the client
	// from walking any further down that particular branch of the client's
	// repository history.  The client may still need to walk down other
	// branches, sending have lines for those, until the server has a
	// complete cut across the DAG, or the client has said "done".
	//
	// Without multi_ack, a client sends have lines in --date-order until
	// the server has found a common base.  That means the client will send
	// have lines that are already known by the server to be common, because
	// they overlap in time with another branch that the server hasn't found
	// a common base on yet.
	//
	// For example suppose the client has commits in caps that the server
	// doesn't and the server has commits in lower case that the client
	// doesn't, as in the following diagram:
	//
	//        +---- u ---------------------- x
	//       /              +----- y
	//      /              /
	//     a -- b -- c -- d -- E -- F
	//        \
	// 	+--- Q -- R -- S
	//
	// If the client wants x,y and starts out by saying have F,S, the server
	// doesn't know what F,S is.  Eventually the client says "have d" and
	// the server sends "ACK d continue" to let the client know to stop
	// walking down that line (so don't send c-b-a), but it's not done yet,
	// it needs a base for x. The client keeps going with S-R-Q, until a
	// gets reached, at which point the server has a clear base and it all
	// ends.
	//
	// Without multi_ack the client would have sent that c-b-a chain anyway,
	// interleaved with S-R-Q.
	MultiACK Capability = "multi_ack"
	// MultiACKDetailed is an extension of multi_ack that permits client to
	// better understand the server's in-memory state.
	MultiACKDetailed Capability = "multi_ack_detailed"
	// NoDone should only be used with the smart HTTP protocol. If
	// multi_ack_detailed and no-done are both present, then the sender is
	// free to immediately send a pack following its first "ACK obj-id ready"
	// message.
	//
	// Without no-done in the smart HTTP protocol, the server session would
	// end and the client has to make another trip to send "done" before
	// the server can send the pack. no-done removes the last round and
	// thus slightly reduces latency.
	NoDone Capability = "no-done"
	// ThinPack is one with deltas which reference base objects not
	// contained within the pack (but are known to exist at the receiving
	// end). This can reduce the network traffic significantly, but it
	// requires the receiving end to know how to "thicken" these packs by
	// adding the missing bases to the pack.
	//
	// The upload-pack server advertises 'thin-pack' when it can generate
	// and send a thin pack. A client requests the 'thin-pack' capability
	// when it understands how to "thicken" it, notifying the server that
	// it can receive such a pack. A client MUST NOT request the
	// 'thin-pack' capability if it cannot turn a thin pack into a
	// self-contained pack.
	//
	// Receive-pack, on the other hand, is assumed by default to be able to
	// handle thin packs, but can ask the client not to use the feature by
	// advertising the 'no-thin' capability. A client MUST NOT send a thin
	// pack if the server advertises the 'no-thin' capability.
	//
	// The reasons for this asymmetry are historical. The receive-pack
	// program did not exist until after the invention of thin packs, so
	// historically the reference implementation of receive-pack always
	// understood thin packs. Adding 'no-thin' later allowed receive-pack
	// to disable the feature in a backwards-compatible manner.
	ThinPack Capability = "thin-pack"
	// Sideband means that server can send, and client understand multiplexed
	// progress reports and error info interleaved with the packfile itself.
	//
	// These two options are mutually exclusive. A modern client always
	// favors Sideband64k.
	//
	// Either mode indicates that the packfile data will be streamed broken
	// up into packets of up to either 1000 bytes in the case of 'side_band',
	// or 65520 bytes in the case of 'side_band_64k'. Each packet is made up
	// of a leading 4-byte pkt-line length of how much data is in the packet,
	// followed by a 1-byte stream code, followed by the actual data.
	//
	// The stream code can be one of:
	//
	//  1 - pack data
	//  2 - progress messages
	//  3 - fatal error message just before stream aborts
	//
	// The "side-band-64k" capability came about as a way for newer clients
	// that can handle much larger packets to request packets that are
	// actually crammed nearly full, while maintaining backward compatibility
	// for the older clients.
	//
	// Further, with side-band and its up to 1000-byte messages, it's actually
	// 999 bytes of payload and 1 byte for the stream code. With side-band-64k,
	// same deal, you have up to 65519 bytes of data and 1 byte for the stream
	// code.
	//
	// The client MUST send only maximum of one of "side-band" and "side-
	// band-64k".  Server MUST diagnose it as an error if client requests
	// both.
	Sideband    Capability = "side-band"
	Sideband64k Capability = "side-band-64k"
	// OFSDelta server can send, and client understand PACKv2 with delta
	// referring to its base by position in pack rather than by an obj-id. That
	// is, they can send/read OBJ_OFS_DELTA (aka type 6) in a packfile.
	OFSDelta Capability = "ofs-delta"
	// Agent the server may optionally send this capability to notify the client
	// that the server is running version `X`. The client may optionally return
	// its own agent string by responding with an `agent=Y` capability (but it
	// MUST NOT do so if the server did not mention the agent capability). The
	// `X` and `Y` strings may contain any printable ASCII characters except
	// space (i.e., the byte range 32 < x < 127), and are typically of the form
	// "package/version" (e.g., "git/1.8.3.1"). The agent strings are purely
	// informative for statistics and debugging purposes, and MUST NOT be used
	// to programmatically assume the presence or absence of particular features.
	Agent Capability = "agent"
	// Shallow capability adds "deepen", "shallow" and "unshallow" commands to
	// the  fetch-pack/upload-pack protocol so clients can request shallow
	// clones.
	Shallow Capability = "shallow"
	// DeepenSince adds "deepen-since" command to fetch-pack/upload-pack
	// protocol so the client can request shallow clones that are cut at a
	// specific time, instead of depth. Internally it's equivalent of doing
	// "rev-list --max-age=<timestamp>" on the server side. "deepen-since"
	// cannot be used with "deepen".
	DeepenSince Capability = "deepen-since"
	// DeepenNot adds "deepen-not" command to fetch-pack/upload-pack
	// protocol so the client can request shallow clones that are cut at a
	// specific revision, instead of depth. Internally it's equivalent of
	// doing "rev-list --not <rev>" on the server side. "deepen-not"
	// cannot be used with "deepen", but can be used with "deepen-since".
	DeepenNot Capability = "deepen-not"
	// DeepenRelative if this capability is requested by the client, the
	// semantics of "deepen" command is changed. The "depth" argument is the
	// depth from the current shallow boundary, instead of the depth from
	// remote refs.
	DeepenRelative Capability = "deepen-relative"
	// NoProgress the client was started with "git clone -q" or something, and
	// doesn't want that side band 2. Basically the client just says "I do not
	// wish to receive stream 2 on sideband, so do not send it to me, and if
	// you did, I will drop it on the floor anyway".  However, the sideband
	// channel 3 is still used for error responses.
	NoProgress Capability = "no-progress"
	// IncludeTag capability is about sending annotated tags if we are
	// sending objects they point to.  If we pack an object to the client, and
	// a tag object points exactly at that object, we pack the tag object too.
	// In general this allows a client to get all new annotated tags when it
	// fetches a branch, in a single network connection.
	//
	// Clients MAY always send include-tag, hardcoding it into a request when
	// the server advertises this capability. The decision for a client to
	// request include-tag only has to do with the client's desires for tag
	// data, whether or not a server had advertised objects in the
	// refs/tags/* namespace.
	//
	// Servers MUST pack the tags if their referrant is packed and the client
	// has requested include-tags.
	//
	// Clients MUST be prepared for the case where a server has ignored
	// include-tag and has not actually sent tags in the pack.  In such
	// cases the client SHOULD issue a subsequent fetch to acquire the tags
	// that include-tag would have otherwise given the client.
	//
	// The server SHOULD send include-tag, if it supports it, regardless
	// of whether or not there are tags available.
	IncludeTag Capability = "include-tag"
	// ReportStatus the receive-pack process can receive a 'report-status'
	// capability, which tells it that the client wants a report of what
	// happened after a packfile upload and reference update. If the pushing
	// client requests this capability, after unpacking and updating references
	// the server will respond with whether the packfile unpacked successfully
	// and if each reference was updated successfully. If any of those were not
	// successful, it will send back an error message.  See pack-protocol.txt
	// for example messages.
	ReportStatus Capability = "report-status"
	// DeleteRefs If the server sends back this capability, it means that
	// it is capable of accepting a zero-id value as the target
	// value of a reference update.  It is not sent back by the client, it
	// simply informs the client that it can be sent zero-id values
	// to delete references
	DeleteRefs Capability = "delete-refs"
	// Quiet If the receive-pack server advertises this capability, it is
	// capable of silencing human-readable progress output which otherwise may
	// be shown when processing the received pack. A send-pack client should
	// respond with the 'quiet' capability to suppress server-side progress
	// reporting if the local progress reporting is also being suppressed
	// (e.g., via `push -q`, or if stderr does not go to a tty).
	Quiet Capability = "quiet"
	// Atomic If the server sends this capability it is capable of accepting
	// atomic pushes. If the pushing client requests this capability, the server
	// will update the refs in one atomic transaction. Either all refs are
	// updated or none.
	Atomic Capability = "atomic"
	// PushOptions If the server sends this capability it is able to accept
	// push options after the update commands have been sent, but before the
	// packfile is streamed. If the pushing client requests this capability,
	// the server will pass the options to the pre- and post- receive hooks
	// that process this push request.
	PushOptions Capability = "push-options"
	// AllowTipSHA1InWant if the upload-pack server advertises this capability,
	// fetch-pack may send "want" lines with SHA-1s that exist at the server but
	// are not advertised by upload-pack.
	AllowTipSHA1InWant Capability = "allow-tip-sha1-in-want"
	// AllowReachableSHA1InWant if the upload-pack server advertises this
	// capability, fetch-pack may send "want" lines with SHA-1s that exist at
	// the server but are not advertised by upload-pack.
	AllowReachableSHA1InWant Capability = "allow-reachable-sha1-in-want"
	// PushCert the receive-pack server that advertises this capability is
	// willing to accept a signed push certificate, and asks the <nonce> to be
	// included in the push certificate.  A send-pack client MUST NOT
	// send a push-cert packet unless the receive-pack server advertises
	// this capability.
	PushCert Capability = "push-cert"
	// SymRef symbolic reference support for better negotiation.
	SymRef Capability = "symref"
)

const DefaultAgent = "go-git/4.x"

var known = map[Capability]bool{
	MultiACK: true, MultiACKDetailed: true, NoDone: true, ThinPack: true,
	Sideband: true, Sideband64k: true, OFSDelta: true, Agent: true,
	Shallow: true, DeepenSince: true, DeepenNot: true, DeepenRelative: true,
	NoProgress: true, IncludeTag: true, ReportStatus: true, DeleteRefs: true,
	Quiet: true, Atomic: true, PushOptions: true, AllowTipSHA1InWant: true,
	AllowReachableSHA1InWant: true, PushCert: true, SymRef: true,
}

var requiresArgument = map[Capability]bool{
	Agent: true, PushCert: true, SymRef: true,
}

var multipleArgument = map[Capability]bool{
	SymRef: true,
}
