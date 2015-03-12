go-xdr
======

[![Build Status](https://travis-ci.org/davecgh/go-xdr.png?branch=master)]
(https://travis-ci.org/davecgh/go-xdr) [![Coverage Status]
(https://coveralls.io/repos/davecgh/go-xdr/badge.png?branch=master)]
(https://coveralls.io/r/davecgh/go-xdr?branch=master)

Go-xdr implements the data representation portion of the External Data
Representation (XDR) standard protocol as specified in RFC 4506 (obsoletes RFC
1832 and RFC 1014) in Pure Go (Golang).  A comprehensive suite of tests are
provided to ensure proper functionality.  It is licensed under the liberal ISC
license, so it may be used in open source or commercial projects.

NOTE: Version 1 of this package is still available via the
github.com/davecgh/go-xdr/xdr import path to avoid breaking existing clients.  However, it is highly recommended that all old clients upgrade to version 2
and all new clients use version 2.  In addition to some speed optimizations,
version 2 has been been updated to work with standard the io.Reader and
io.Writer interfaces instead of raw byte slices.  This allows it to by much more
flexible and work directly with files, network connections, etc.

## Documentation

[![GoDoc](https://godoc.org/github.com/davecgh/go-xdr/xdr2?status.png)]
(http://godoc.org/github.com/davecgh/go-xdr/xdr2)

Full `go doc` style documentation for the project can be viewed online without
installing this package by using the excellent GoDoc site here:
http://godoc.org/github.com/davecgh/go-xdr/xdr2

You can also view the documentation locally once the package is installed with
the `godoc` tool by running `godoc -http=":6060"` and pointing your browser to
http://localhost:6060/pkg/github.com/davecgh/go-xdr/xdr2/

## Installation

```bash
$ go get github.com/davecgh/go-xdr/xdr2
```

## Sample Decode Program

```Go
package main

import (
	"bytes"
    "fmt"

    "github.com/davecgh/go-xdr/xdr2"
)

func main() {
	// Hypothetical image header format.
	type ImageHeader struct {
		Signature   [3]byte
		Version     uint32
		IsGrayscale bool
		NumSections uint32
	}

	// XDR encoded data described by the above structure.  Typically this would
	// be read from a file or across the network, but use a manual byte array
	// here as an example.
	encodedData := []byte{
		0xAB, 0xCD, 0xEF, 0x00, // Signature
		0x00, 0x00, 0x00, 0x02, // Version
		0x00, 0x00, 0x00, 0x01, // IsGrayscale
		0x00, 0x00, 0x00, 0x0A, // NumSections
	}

	// Declare a variable to provide Unmarshal with a concrete type and instance
	// to decode into.
	var h ImageHeader
	bytesRead, err := xdr.Unmarshal(bytes.NewReader(encodedData), &h)
	if err != nil {
		fmt.Println(err)
		return
	}
  
	fmt.Println("bytes read:", bytesRead)
	fmt.Printf("h: %+v", h)
}
```

The struct instance, `h`, will then contain the following values:

```Go
h.Signature = [3]byte{0xAB, 0xCD, 0xEF}
h.Version = 2
h.IsGrayscale = true
h.NumSections = 10
```

## Sample Encode Program

```Go
package main

import (
	"bytes"
    "fmt"

    "github.com/davecgh/go-xdr/xdr2"
)

func main() {
	// Hypothetical image header format.
	type ImageHeader struct {
		Signature   [3]byte
		Version     uint32
		IsGrayscale bool
		NumSections uint32
	}

	// Sample image header data.
	h := ImageHeader{[3]byte{0xAB, 0xCD, 0xEF}, 2, true, 10}

	// Use marshal to automatically determine the appropriate underlying XDR
	// types and encode.
	var w bytes.Buffer
	bytesWritten, err := xdr.Marshal(&w, &h)
	if err != nil {
		fmt.Println(err)
		return
	}

	encodedData := w.Bytes()
	fmt.Println("bytes written:", bytesWritten)
	fmt.Println("encoded data:", encodedData)
}
```

The result, `encodedData`, will then contain the following XDR encoded byte
sequence:

```
0xAB, 0xCD, 0xEF, 0x00,
0x00, 0x00, 0x00, 0x02,
0x00, 0x00, 0x00, 0x01,
0x00, 0x00, 0x00, 0x0A,
```

## License

Go-xdr is licensed under the liberal ISC License.
