// Package recordio implements the Mesos variant of RecordIO framing, whereby
// each record is prefixed by a line that indicates the length of the record in
// decimal ASCII. The bytes of the record immediately follow the length-line.
// Zero-length records are allowed.
package recordio
