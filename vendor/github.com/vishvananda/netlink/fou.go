package netlink

import (
	"errors"
)

var (
	// ErrAttrHeaderTruncated is returned when a netlink attribute's header is
	// truncated.
	ErrAttrHeaderTruncated = errors.New("attribute header truncated")
	// ErrAttrBodyTruncated is returned when a netlink attribute's body is
	// truncated.
	ErrAttrBodyTruncated = errors.New("attribute body truncated")
)

type Fou struct {
	Family    int
	Port      int
	Protocol  int
	EncapType int
}
