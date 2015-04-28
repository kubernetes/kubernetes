// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Anatomy of a WAL file

WAL file
	A sequence of packets

WAL packet, parts in slice notation
	[0:4],   4 bytes:        N uint32        // network byte order
	[4:4+N], N bytes:        payload []byte  // gb encoded scalars

Packets, including the 4 byte 'size' prefix, MUST BE padded to size == 0 (mod
16). The values of the padding bytes MUST BE zero.

Encoded scalars first item is a packet type number (packet tag). The meaning of
any other item(s) of the payload depends on the packet tag.

Packet definitions

	{wpt00Header int, typ int, s string}
		typ:	Must be zero (ACIDFiler0 file).
		s:	Any comment string, empty string is okay.

		This packet must be present only once - as the first packet of
		a WAL file.

	{wpt00WriteData int, b []byte, off int64}
		Write data (WriteAt(b, off)).

	{wpt00Checkpoint int, sz int64}
		Checkpoint (Truncate(sz)).

		This packet must be present only once - as the last packet of
		a WAL file.

*/

package lldb

//TODO optimize bitfiler/wal/2pc data above final size
