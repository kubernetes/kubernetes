// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

/*

WIP: Package falloc provides allocation/deallocation of space within a
file/store (WIP, unstable API).

Overall structure:
 File == n blocks.
 Block == n atoms.
 Atom == 16 bytes.

x6..x0 == least significant 7 bytes of a 64 bit integer, highest (7th) byte is
0 and is not stored in the file.

Block first byte

Aka block type tag.

------------------------------------------------------------------------------

0xFF: Free atom (free block of size 1).
 +------++---------++---------++------+
 |  0   ||  1...7  || 8...14  ||  15  |
 +------++---------++---------++------+
 | 0xFF || p6...p0 || n6...n0 || 0xFF |
 +------++---------++---------++------+

Link to the previous free block (atom addressed) is p6...p0, next dtto in
n6...n0.  Doubly linked lists of "compatible" free blocks allows for free space
reclaiming and merging.  "Compatible" == of size at least some K. Heads of all
such lists are organized per K or intervals of Ks elsewhere.

------------------------------------------------------------------------------

0xFE: Free block, size == s6...s0 atoms.
 +------++---------++---------++---------++--
 |  +0  ||  1...7  || 8...14  || 15...21 || 22...16*size-1
 +------++---------++---------++---------++--
 | 0xFE || p6...p0 || n6...n0 || s6...s0 || ...
 +------++---------++---------++---------++--

Prev and next links as in the 0xFF first byte case.  End of this block - see
"Block last byte": 0xFE bellow. Data between == undefined.

------------------------------------------------------------------------------

0xFD:  Relocated block.
 +------++---------++-----------++------+
 |  0   ||  1...7  ||  8...14   ||  15  |
 +------++---------++-----------++------+
 | 0xFD || r6...r0 || undefined || 0x00 | // == used block
 +------++---------++-----------++------+

Relocation link is r6..r0 == atom address. Relocations MUST NOT chain and MUST
point to a "content" block, i.e. one with the first byte in 0x00...0xFC.

Relocated block allows to permanently assign a handle/file pointer ("atom"
address) to some content and resize the content anytime afterwards w/o having
to update all the possible existing references to the original handle.

------------------------------------------------------------------------------

0xFC: Used long block.
 +------++---------++--------------------++---------+---+
 |  0   ||  1...2  ||      3...N+2       ||         |   |
 +------++---------++--------------------++---------+---+
 | 0xFC || n1...n0 || N bytes of content || padding | Z |
 +------++---------++--------------------++---------+---+

This block type is used for content of length in N == 238...61680 bytes. N is
encoded as a 2 byte unsigned integer n1..n0 in network byte order. Values
bellow 238 are reserved, those content lengths are to be carried by the
0x00..0xFB block types.

 1. n in 0x00EE...0xF0F0 is used for content under the same rules
    as in the 0x01..0xED type.

 2. If the last byte of the content is not the last byte of an atom then
    the last byte of the block is 0x00.

 3. If the last byte of the content IS the last byte of an atom:

   3.1 If the last byte of content is in 0x00..0xFD then everything is OK.

   3.2 If the last byte of content is 0xFE or 0xFF then the escape
       via n > 0xF0F0 MUST be used AND the block's last byte is 0x00 or 0x01,
       meaning value 0xFE and 0xFF respectively.

 4. n in 0xF0F1...0xFFFF is like the escaped 0xEE..0xFB block.
    N == 13 + 16(n - 0xF0F1).

Discussion of the padding and Z fields - see the 0x01..0xED block type.

------------------------------------------------------------------------------

0xEE...0xFB: Used escaped short block.
 +---++----------------------++---+
 | 0 ||        1...N-1       ||   |
 +---++----------------------++---+
 | X || N-1 bytes of content || Z |
 +---++----------------------++---+

N == 15 + 16(X - 0xEE). Z is the content last byte encoded as follows.

case Z == 0x00:	The last byte of content is 0xFE

case Z == 0x01:	The last byte of content is 0xFF

------------------------------------------------------------------------------

0x01...0xED: Used short block.
 +---++--------------------++---------+---+
 | 0 ||        1...N       ||         |   |
 +---++--------------------++---------+---+
 | N || N bytes of content || padding | Z |
 +---++--------------------++---------+---+

This block type is used for content of length in 1...237 bytes.  The value of
the "padding" field, if of non zero length, is undefined.

If the last byte of content is the last byte of an atom (== its file byte
offset & 0xF == 0xF) then such last byte MUST be in 0x00...0xFD.

If the last byte of content is the last byte of an atom AND the last byte of
content is 0xFE or 0xFF then the short escape block type (0xEE...0xFB) MUST be
used.

If the last byte of content is not the last byte of an atom, then the last byte
of such block, i.e. the Z field, which is also a last byte of some atom, MUST
be 0x00 (i.e. the used block marker).  Other "tail" values are reserved.

------------------------------------------------------------------------------

0x00: Used empty block.
 +------++-----------++------+
 |  0   ||  1...14   ||  15  |
 +------++-----------++------+
 | 0x00 || undefined || 0x00 | // == used block, other "tail" values reserved.
 +------++-----------++------+

All of the rules for 0x01..0xED applies. Depicted only for its different
semantics (e.g. an allocated [existing] string but with length of zero).

==============================================================================

Block last byte

------------------------------------------------------------------------------

0xFF: Free atom. Layout - see "Block first byte": FF.

------------------------------------------------------------------------------

0xFE: Free block, size n atoms. Preceding 7 bytes == size (s6...s0) of the free
block in atoms, network byte order
   --++---------++------+
     || -8...-2 ||  -1  |
   --++---------++------+
 ... || s6...s0 || 0xFE | <- block's last byte
   --++---------++------+

Layout at start of this block - see "Block first byte": FE.

------------------------------------------------------------------------------

0x00...0xFD: Used (non free) block.

==============================================================================

Free lists table

The free lists table content is stored in the standard layout of a used block.

A table item is a 7 byte size field followed by a 7 byte atom address field
(both in network byte order), thus every item is 14 contiguous bytes. The
item's address field is pointing to a free block.  The size field determines
the minimal size (in atoms) of free blocks on that list.

The free list table is n above items, thus the content has 14n bytes. Note that
the largest block content is 61680 bytes and as there are 14 bytes per table
item, so the table is limited to at most 4405 entries.

Items in the table do not have to be sorted according to their size field values.

No two items can have the same value of the size field.

When freeing blocks, the block MUST be linked into an item list with the
highest possible size field, which is less or equal to the number of atoms in
the new free block.

When freeing a block, the block MUST be first merged with any adjacent free
blocks (thus possibly creating a bigger free block) using information derived
from the adjacent blocks first and last bytes. Such merged free blocks MUST be
removed from their original doubly linked lists. Afterwards the new bigger free
block is put to the free list table in the appropriate item.

Items with address field == 0 are legal. Such item is a placeholder for a empty
list of free blocks of the item's size.

Items with size field == 0 are legal. Such item is a placeholder, used e.g. to
avoid further reallocations/redirecting of the free lists table.

The largest possible allocation request (for content length 61680 bytes) is
0xF10 (3856) atoms.  All free blocks of this or bigger size are presumably put
into a single table item with the size 3856. It may be useful to additionally
have a free lists table item which links free blocks of some bigger size (say
1M+) and then use the OS sparse file support (if present) to save the physical
space used by such free blocks.

Smaller (<3856 atoms) free blocks can be organized exactly (every distinct size
has its table item) or the sizes can run using other schema like e.g. "1, 2,
4, 8, ..." (powers of 2) or "1, 2, 3, 5, 8, 13, ..." (the Fibonacci sequence)
or they may be fine tuned to a specific usage pattern.

==============================================================================

Header

The first block of a file (atom address == file offset == 0) is the file header.
The header block has the standard layout of a used short non escaped block.

Special conditions apply: The header block and its content MUST be like this:

 +------+---------+---------+------+
 |  0   |  1...7  | 8...14  |  15  |
 +------+---------+---------+------+
 | 0x0F | m6...m0 | f6...f0 | FLTT |
 +------+---------+---------+------+

m6..m0 is a "magic" value 0xF1C1A1FE51B1E.

f6...f0 is the atom address of the free lists table (discussed elsewhere).
If f6...f0 == 0x00 the there is no free lists table (yet).

FLTT describes the type of the Free List Table. Currently defined values:

------------------------------------------------------------------------------

FLTT == 0: Free List Table is fixed at atom address 2. It has a fixed size for 3856 entries
for free list of size 1..3855 atoms and the last is for the list of free block >= 3856 atoms.
*/
package falloc

const (
	INVALID_HANDLE = Handle(-1)
)
