/*
Package xt implements dedicated types for (some) of the "Info" payload in Match
and Target expressions that bridge between the nftables and xtables worlds.

Bridging between the more unified world of nftables and the slightly
heterogenous world of xtables comes with some caveats. Unmarshalling the
extension/translation information in Match and Target expressions requires
information about the table family the information belongs to, as well as type
and type revision information. In consequence, unmarshalling the Match and
Target Info field payloads often (but not necessarily always) require the table
family and revision information, so it gets passed to the type-specific
unmarshallers.

To complicate things more, even marshalling requires knowledge about the
enclosing table family. The NatRange/NatRange2 types are an example, where it is
necessary to differentiate between IPv4 and IPv6 address marshalling. Due to
Go's net.IP habit to normally store IPv4 addresses as IPv4-compatible IPv6
addresses (see also RFC 4291, section 2.5.5.1) marshalling must be handled
differently in the context of an IPv6 table compared to an IPv4 table. In an
IPv4 table, an IPv4-compatible IPv6 address must be marshalled as a 32bit
address, whereas in an IPv6 table the IPv4 address must be marshalled as an
128bit IPv4-compatible IPv6 address. Not relying on heuristics here we avoid
behavior unexpected and most probably unknown to our API users. The net.IP habit
of storing IPv4 addresses in two different storage formats is already a source
for trouble, especially when comparing net.IPs from different Go module sources.
We won't add to this confusion. (...or maybe we can, because of it?)

An important property of all types of Info extension/translation payloads is
that their marshalling and unmarshalling doesn't follow netlink's TLV
(tag-length-value) architecture. Instead, Info payloads a basically plain binary
blobs of their respective type-specific data structures, so host
platform/architecture alignment and data type sizes apply. The alignedbuff
package implements the different required data types alignments.

Please note that Info payloads are always padded at their end to the next uint64
alignment. Kernel code is checking for the padded payload size and will reject
payloads not correctly padded at their ends.

Most of the time, we find explifcitly sized (unsigned integer) data types.
However, there are notable exceptions where "unsigned int" is used: on 64bit
platforms this mostly translates into 32bit(!). This differs from Go mapping
uint to uint64 instead. This package currently clamps its mapping of C's
"unsigned int" to Go's uint32 for marshalling and unmarshalling. If in the
future 128bit platforms with a differently sized C unsigned int should come into
production, then the alignedbuff package will need to be adapted accordingly, as
it abstracts away this data type handling.
*/
package xt
