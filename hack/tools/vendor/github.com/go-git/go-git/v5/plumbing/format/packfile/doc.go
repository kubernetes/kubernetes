// Package packfile implements encoding and decoding of packfile format.
//
//  == pack-*.pack files have the following format:
//
//    - A header appears at the beginning and consists of the following:
//
//      4-byte signature:
//          The signature is: {'P', 'A', 'C', 'K'}
//
//      4-byte version number (network byte order):
//          GIT currently accepts version number 2 or 3 but
//          generates version 2 only.
//
//      4-byte number of objects contained in the pack (network byte order)
//
//      Observation: we cannot have more than 4G versions ;-) and
//      more than 4G objects in a pack.
//
//    - The header is followed by number of object entries, each of
//      which looks like this:
//
//      (undeltified representation)
//      n-byte type and length (3-bit type, (n-1)*7+4-bit length)
//      compressed data
//
//      (deltified representation)
//      n-byte type and length (3-bit type, (n-1)*7+4-bit length)
//      20-byte base object name
//      compressed delta data
//
//      Observation: length of each object is encoded in a variable
//      length format and is not constrained to 32-bit or anything.
//
//   - The trailer records 20-byte SHA1 checksum of all of the above.
//
//
// Source:
// https://www.kernel.org/pub/software/scm/git/docs/v1.7.5/technical/pack-protocol.txt
package packfile
