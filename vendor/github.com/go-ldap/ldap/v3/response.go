package ldap

import (
	"context"
	"errors"
	"fmt"

	ber "github.com/go-asn1-ber/asn1-ber"
)

// Response defines an interface to get data from an LDAP server
type Response interface {
	Entry() *Entry
	Referral() string
	Controls() []Control
	Err() error
	Next() bool
}

type searchResponse struct {
	conn *Conn
	ch   chan *SearchSingleResult

	entry    *Entry
	referral string
	controls []Control
	err      error
}

// Entry returns an entry from the given search request
func (r *searchResponse) Entry() *Entry {
	return r.entry
}

// Referral returns a referral from the given search request
func (r *searchResponse) Referral() string {
	return r.referral
}

// Controls returns controls from the given search request
func (r *searchResponse) Controls() []Control {
	return r.controls
}

// Err returns an error when the given search request was failed
func (r *searchResponse) Err() error {
	return r.err
}

// Next returns whether next data exist or not
func (r *searchResponse) Next() bool {
	res, ok := <-r.ch
	if !ok {
		return false
	}
	if res == nil {
		return false
	}
	r.err = res.Error
	if r.err != nil {
		return false
	}
	r.entry = res.Entry
	r.referral = res.Referral
	r.controls = res.Controls
	return true
}

func (r *searchResponse) start(ctx context.Context, searchRequest *SearchRequest) {
	go func() {
		defer func() {
			close(r.ch)
			if err := recover(); err != nil {
				r.conn.err = fmt.Errorf("ldap: recovered panic in searchResponse: %v", err)
			}
		}()

		if r.conn.IsClosing() {
			return
		}

		packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
		packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, r.conn.nextMessageID(), "MessageID"))
		// encode search request
		err := searchRequest.appendTo(packet)
		if err != nil {
			r.ch <- &SearchSingleResult{Error: err}
			return
		}
		r.conn.Debug.PrintPacket(packet)

		msgCtx, err := r.conn.sendMessage(packet)
		if err != nil {
			r.ch <- &SearchSingleResult{Error: err}
			return
		}
		defer r.conn.finishMessage(msgCtx)

		foundSearchSingleResultDone := false
		for !foundSearchSingleResultDone {
			r.conn.Debug.Printf("%d: waiting for response", msgCtx.id)
			select {
			case <-ctx.Done():
				r.conn.Debug.Printf("%d: %s", msgCtx.id, ctx.Err().Error())
				return
			case packetResponse, ok := <-msgCtx.responses:
				if !ok {
					err := NewError(ErrorNetwork, errors.New("ldap: response channel closed"))
					r.ch <- &SearchSingleResult{Error: err}
					return
				}
				packet, err = packetResponse.ReadPacket()
				r.conn.Debug.Printf("%d: got response %p", msgCtx.id, packet)
				if err != nil {
					r.ch <- &SearchSingleResult{Error: err}
					return
				}

				if r.conn.Debug {
					if err := addLDAPDescriptions(packet); err != nil {
						r.ch <- &SearchSingleResult{Error: err}
						return
					}
					ber.PrintPacket(packet)
				}

				switch packet.Children[1].Tag {
				case ApplicationSearchResultEntry:
					result := &SearchSingleResult{
						Entry: &Entry{
							DN:         packet.Children[1].Children[0].Value.(string),
							Attributes: unpackAttributes(packet.Children[1].Children[1].Children),
						},
					}
					if len(packet.Children) != 3 {
						r.ch <- result
						continue
					}
					decoded, err := DecodeControl(packet.Children[2].Children[0])
					if err != nil {
						werr := fmt.Errorf("failed to decode search result entry: %w", err)
						result.Error = werr
						r.ch <- result
						return
					}
					result.Controls = append(result.Controls, decoded)
					r.ch <- result

				case ApplicationSearchResultDone:
					if err := GetLDAPError(packet); err != nil {
						r.ch <- &SearchSingleResult{Error: err}
						return
					}
					if len(packet.Children) == 3 {
						result := &SearchSingleResult{}
						for _, child := range packet.Children[2].Children {
							decodedChild, err := DecodeControl(child)
							if err != nil {
								werr := fmt.Errorf("failed to decode child control: %w", err)
								r.ch <- &SearchSingleResult{Error: werr}
								return
							}
							result.Controls = append(result.Controls, decodedChild)
						}
						r.ch <- result
					}
					foundSearchSingleResultDone = true

				case ApplicationSearchResultReference:
					ref := packet.Children[1].Children[0].Value.(string)
					r.ch <- &SearchSingleResult{Referral: ref}

				case ApplicationIntermediateResponse:
					decoded, err := DecodeControl(packet.Children[1])
					if err != nil {
						werr := fmt.Errorf("failed to decode intermediate response: %w", err)
						r.ch <- &SearchSingleResult{Error: werr}
						return
					}
					result := &SearchSingleResult{}
					result.Controls = append(result.Controls, decoded)
					r.ch <- result

				default:
					err := fmt.Errorf("unknown tag: %d", packet.Children[1].Tag)
					r.ch <- &SearchSingleResult{Error: err}
					return
				}
			}
		}
		r.conn.Debug.Printf("%d: returning", msgCtx.id)
	}()
}

func newSearchResponse(conn *Conn, bufferSize int) *searchResponse {
	var ch chan *SearchSingleResult
	if bufferSize > 0 {
		ch = make(chan *SearchSingleResult, bufferSize)
	} else {
		ch = make(chan *SearchSingleResult)
	}
	return &searchResponse{
		conn: conn,
		ch:   ch,
	}
}
