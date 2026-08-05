package xt

import (
	"bytes"
	"fmt"
)

// CommentSize is the fixed size of a comment info xt blob, see:
// https://elixir.bootlin.com/linux/v6.8.7/source/include/uapi/linux/netfilter/xt_comment.h#L5
const CommentSize = 256

// Comment gets marshalled and unmarshalled as a fixed-sized char array, filled
// with zeros as necessary, see:
// https://elixir.bootlin.com/linux/v6.8.7/source/include/uapi/linux/netfilter/xt_comment.h#L7
type Comment string

func (c *Comment) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	if len(*c) >= CommentSize {
		return nil, fmt.Errorf("comment must be less than %d bytes, got %d bytes",
			CommentSize, len(*c))
	}
	data := make([]byte, CommentSize)
	copy(data, []byte(*c))
	return data, nil
}

func (c *Comment) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	if len(data) != CommentSize {
		return fmt.Errorf("malformed comment: got %d bytes, expected exactly %d bytes",
			len(data), CommentSize)
	}
	*c = Comment(bytes.TrimRight(data, "\x00"))
	return nil
}
