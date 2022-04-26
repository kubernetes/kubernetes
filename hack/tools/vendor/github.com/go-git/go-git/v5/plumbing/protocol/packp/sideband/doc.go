// Package sideband implements a sideband mutiplex/demultiplexer
package sideband

// If 'side-band' or 'side-band-64k' capabilities have been specified by
// the client, the server will send the packfile data multiplexed.
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
