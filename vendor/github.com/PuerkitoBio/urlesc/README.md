urlesc [![Build Status](https://travis-ci.org/opennota/urlesc.png?branch=master)](https://travis-ci.org/opennota/urlesc) [![GoDoc](http://godoc.org/github.com/opennota/urlesc?status.svg)](http://godoc.org/github.com/opennota/urlesc)
======

Package urlesc implements query escaping as per RFC 3986.

It contains some parts of the net/url package, modified so as to allow
some reserved characters incorrectly escaped by net/url (see [issue 5684](https://github.com/golang/go/issues/5684)).

## Install

    go get github.com/opennota/urlesc

## License

MIT

