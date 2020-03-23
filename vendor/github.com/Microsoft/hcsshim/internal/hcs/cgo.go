package hcs

import "C"

// This import is needed to make the library compile as CGO because HCSSHIM
// only works with CGO due to callbacks from HCS comming back from a C thread
// which is not supported without CGO. See https://github.com/golang/go/issues/10973
