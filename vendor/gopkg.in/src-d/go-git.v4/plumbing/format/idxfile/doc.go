// Package idxfile implements encoding and decoding of packfile idx files.
//
//  == Original (version 1) pack-*.idx files have the following format:
//
//    - The header consists of 256 4-byte network byte order
//      integers.  N-th entry of this table records the number of
//      objects in the corresponding pack, the first byte of whose
//      object name is less than or equal to N.  This is called the
//      'first-level fan-out' table.
//
//    - The header is followed by sorted 24-byte entries, one entry
//      per object in the pack.  Each entry is:
//
//     4-byte network byte order integer, recording where the
//     object is stored in the packfile as the offset from the
//     beginning.
//
//     20-byte object name.
//
//   - The file is concluded with a trailer:
//
//     A copy of the 20-byte SHA1 checksum at the end of
//     corresponding packfile.
//
//     20-byte SHA1-checksum of all of the above.
//
//   Pack Idx file:
//
//        --  +--------------------------------+
//   fanout   | fanout[0] = 2 (for example)    |-.
//   table    +--------------------------------+ |
//            | fanout[1]                      | |
//            +--------------------------------+ |
//            | fanout[2]                      | |
//            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |
//            | fanout[255] = total objects    |---.
//        --  +--------------------------------+ | |
//   main     | offset                         | | |
//   index    | object name 00XXXXXXXXXXXXXXXX | | |
//   tab      +--------------------------------+ | |
//            | offset                         | | |
//            | object name 00XXXXXXXXXXXXXXXX | | |
//            +--------------------------------+<+ |
//          .-| offset                         |   |
//          | | object name 01XXXXXXXXXXXXXXXX |   |
//          | +--------------------------------+   |
//          | | offset                         |   |
//          | | object name 01XXXXXXXXXXXXXXXX |   |
//          | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   |
//          | | offset                         |   |
//          | | object name FFXXXXXXXXXXXXXXXX |   |
//        --| +--------------------------------+<--+
//  trailer | | packfile checksum              |
//          | +--------------------------------+
//          | | idxfile checksum               |
//          | +--------------------------------+
//          .---------.
//                    |
//  Pack file entry: <+
//
//     packed object header:
//     1-byte size extension bit (MSB)
//           type (next 3 bit)
//           size0 (lower 4-bit)
//         n-byte sizeN (as long as MSB is set, each 7-bit)
//         size0..sizeN form 4+7+7+..+7 bit integer, size0
//         is the least significant part, and sizeN is the
//         most significant part.
//     packed object data:
//         If it is not DELTA, then deflated bytes (the size above
//         is the size before compression).
//     If it is REF_DELTA, then
//       20-byte base object name SHA1 (the size above is the
//         size of the delta data that follows).
//           delta data, deflated.
//     If it is OFS_DELTA, then
//       n-byte offset (see below) interpreted as a negative
//         offset from the type-byte of the header of the
//         ofs-delta entry (the size above is the size of
//         the delta data that follows).
//       delta data, deflated.
//
//     offset encoding:
//       n bytes with MSB set in all but the last one.
//       The offset is then the number constructed by
//       concatenating the lower 7 bit of each byte, and
//       for n >= 2 adding 2^7 + 2^14 + ... + 2^(7*(n-1))
//       to the result.
//
//   == Version 2 pack-*.idx files support packs larger than 4 GiB, and
//      have some other reorganizations.  They have the format:
//
//     - A 4-byte magic number '\377tOc' which is an unreasonable
//       fanout[0] value.
//
//     - A 4-byte version number (= 2)
//
//     - A 256-entry fan-out table just like v1.
//
//     - A table of sorted 20-byte SHA1 object names.  These are
//       packed together without offset values to reduce the cache
//       footprint of the binary search for a specific object name.
//
//     - A table of 4-byte CRC32 values of the packed object data.
//       This is new in v2 so compressed data can be copied directly
//       from pack to pack during repacking without undetected
//       data corruption.
//
//     - A table of 4-byte offset values (in network byte order).
//       These are usually 31-bit pack file offsets, but large
//       offsets are encoded as an index into the next table with
//       the msbit set.
//
//     - A table of 8-byte offset entries (empty for pack files less
//       than 2 GiB).  Pack files are organized with heavily used
//       objects toward the front, so most object references should
//       not need to refer to this table.
//
//     - The same trailer as a v1 pack file:
//
//       A copy of the 20-byte SHA1 checksum at the end of
//       corresponding packfile.
//
//       20-byte SHA1-checksum of all of the above.
//
// Source:
// https://www.kernel.org/pub/software/scm/git/docs/v1.7.5/technical/pack-format.txt
package idxfile
