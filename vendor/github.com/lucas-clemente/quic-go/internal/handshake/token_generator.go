package handshake

import (
	"encoding/asn1"
	"fmt"
	"io"
	"net"
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
)

const (
	tokenPrefixIP byte = iota
	tokenPrefixString
)

// A Token is derived from the client address and can be used to verify the ownership of this address.
type Token struct {
	IsRetryToken bool
	RemoteAddr   string
	SentTime     time.Time
	// only set for retry tokens
	OriginalDestConnectionID protocol.ConnectionID
	RetrySrcConnectionID     protocol.ConnectionID
}

// token is the struct that is used for ASN1 serialization and deserialization
type token struct {
	IsRetryToken             bool
	RemoteAddr               []byte
	Timestamp                int64
	OriginalDestConnectionID []byte
	RetrySrcConnectionID     []byte
}

// A TokenGenerator generates tokens
type TokenGenerator struct {
	tokenProtector tokenProtector
}

// NewTokenGenerator initializes a new TookenGenerator
func NewTokenGenerator(rand io.Reader) (*TokenGenerator, error) {
	tokenProtector, err := newTokenProtector(rand)
	if err != nil {
		return nil, err
	}
	return &TokenGenerator{
		tokenProtector: tokenProtector,
	}, nil
}

// NewRetryToken generates a new token for a Retry for a given source address
func (g *TokenGenerator) NewRetryToken(
	raddr net.Addr,
	origDestConnID protocol.ConnectionID,
	retrySrcConnID protocol.ConnectionID,
) ([]byte, error) {
	data, err := asn1.Marshal(token{
		IsRetryToken:             true,
		RemoteAddr:               encodeRemoteAddr(raddr),
		OriginalDestConnectionID: origDestConnID,
		RetrySrcConnectionID:     retrySrcConnID,
		Timestamp:                time.Now().UnixNano(),
	})
	if err != nil {
		return nil, err
	}
	return g.tokenProtector.NewToken(data)
}

// NewToken generates a new token to be sent in a NEW_TOKEN frame
func (g *TokenGenerator) NewToken(raddr net.Addr) ([]byte, error) {
	data, err := asn1.Marshal(token{
		RemoteAddr: encodeRemoteAddr(raddr),
		Timestamp:  time.Now().UnixNano(),
	})
	if err != nil {
		return nil, err
	}
	return g.tokenProtector.NewToken(data)
}

// DecodeToken decodes a token
func (g *TokenGenerator) DecodeToken(encrypted []byte) (*Token, error) {
	// if the client didn't send any token, DecodeToken will be called with a nil-slice
	if len(encrypted) == 0 {
		return nil, nil
	}

	data, err := g.tokenProtector.DecodeToken(encrypted)
	if err != nil {
		return nil, err
	}
	t := &token{}
	rest, err := asn1.Unmarshal(data, t)
	if err != nil {
		return nil, err
	}
	if len(rest) != 0 {
		return nil, fmt.Errorf("rest when unpacking token: %d", len(rest))
	}
	token := &Token{
		IsRetryToken: t.IsRetryToken,
		RemoteAddr:   decodeRemoteAddr(t.RemoteAddr),
		SentTime:     time.Unix(0, t.Timestamp),
	}
	if t.IsRetryToken {
		token.OriginalDestConnectionID = protocol.ConnectionID(t.OriginalDestConnectionID)
		token.RetrySrcConnectionID = protocol.ConnectionID(t.RetrySrcConnectionID)
	}
	return token, nil
}

// encodeRemoteAddr encodes a remote address such that it can be saved in the token
func encodeRemoteAddr(remoteAddr net.Addr) []byte {
	if udpAddr, ok := remoteAddr.(*net.UDPAddr); ok {
		return append([]byte{tokenPrefixIP}, udpAddr.IP...)
	}
	return append([]byte{tokenPrefixString}, []byte(remoteAddr.String())...)
}

// decodeRemoteAddr decodes the remote address saved in the token
func decodeRemoteAddr(data []byte) string {
	// data will never be empty for a token that we generated.
	// Check it to be on the safe side
	if len(data) == 0 {
		return ""
	}
	if data[0] == tokenPrefixIP {
		return net.IP(data[1:]).String()
	}
	return string(data[1:])
}
