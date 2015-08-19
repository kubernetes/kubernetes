// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// File contains Search functionality
//
// https://tools.ietf.org/html/rfc4511
//
//         SearchRequest ::= [APPLICATION 3] SEQUENCE {
//              baseObject      LDAPDN,
//              scope           ENUMERATED {
//                   baseObject              (0),
//                   singleLevel             (1),
//                   wholeSubtree            (2),
//                   ...  },
//              derefAliases    ENUMERATED {
//                   neverDerefAliases       (0),
//                   derefInSearching        (1),
//                   derefFindingBaseObj     (2),
//                   derefAlways             (3) },
//              sizeLimit       INTEGER (0 ..  maxInt),
//              timeLimit       INTEGER (0 ..  maxInt),
//              typesOnly       BOOLEAN,
//              filter          Filter,
//              attributes      AttributeSelection }
//
//         AttributeSelection ::= SEQUENCE OF selector LDAPString
//                         -- The LDAPString is constrained to
//                         -- <attributeSelector> in Section 4.5.1.8
//
//         Filter ::= CHOICE {
//              and             [0] SET SIZE (1..MAX) OF filter Filter,
//              or              [1] SET SIZE (1..MAX) OF filter Filter,
//              not             [2] Filter,
//              equalityMatch   [3] AttributeValueAssertion,
//              substrings      [4] SubstringFilter,
//              greaterOrEqual  [5] AttributeValueAssertion,
//              lessOrEqual     [6] AttributeValueAssertion,
//              present         [7] AttributeDescription,
//              approxMatch     [8] AttributeValueAssertion,
//              extensibleMatch [9] MatchingRuleAssertion,
//              ...  }
//
//         SubstringFilter ::= SEQUENCE {
//              type           AttributeDescription,
//              substrings     SEQUENCE SIZE (1..MAX) OF substring CHOICE {
//                   initial [0] AssertionValue,  -- can occur at most once
//                   any     [1] AssertionValue,
//                   final   [2] AssertionValue } -- can occur at most once
//              }
//
//         MatchingRuleAssertion ::= SEQUENCE {
//              matchingRule    [1] MatchingRuleId OPTIONAL,
//              type            [2] AttributeDescription OPTIONAL,
//              matchValue      [3] AssertionValue,
//              dnAttributes    [4] BOOLEAN DEFAULT FALSE }
//
//

package ldap

import (
	"errors"
	"fmt"
	"strings"

	"gopkg.in/asn1-ber.v1"
)

const (
	ScopeBaseObject   = 0
	ScopeSingleLevel  = 1
	ScopeWholeSubtree = 2
)

var ScopeMap = map[int]string{
	ScopeBaseObject:   "Base Object",
	ScopeSingleLevel:  "Single Level",
	ScopeWholeSubtree: "Whole Subtree",
}

const (
	NeverDerefAliases   = 0
	DerefInSearching    = 1
	DerefFindingBaseObj = 2
	DerefAlways         = 3
)

var DerefMap = map[int]string{
	NeverDerefAliases:   "NeverDerefAliases",
	DerefInSearching:    "DerefInSearching",
	DerefFindingBaseObj: "DerefFindingBaseObj",
	DerefAlways:         "DerefAlways",
}

type Entry struct {
	DN         string
	Attributes []*EntryAttribute
}

func (e *Entry) GetAttributeValues(attribute string) []string {
	for _, attr := range e.Attributes {
		if attr.Name == attribute {
			return attr.Values
		}
	}
	return []string{}
}

func (e *Entry) GetRawAttributeValues(attribute string) [][]byte {
	for _, attr := range e.Attributes {
		if attr.Name == attribute {
			return attr.ByteValues
		}
	}
	return [][]byte{}
}

func (e *Entry) GetAttributeValue(attribute string) string {
	values := e.GetAttributeValues(attribute)
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

func (e *Entry) GetRawAttributeValue(attribute string) []byte {
	values := e.GetRawAttributeValues(attribute)
	if len(values) == 0 {
		return []byte{}
	}
	return values[0]
}

func (e *Entry) Print() {
	fmt.Printf("DN: %s\n", e.DN)
	for _, attr := range e.Attributes {
		attr.Print()
	}
}

func (e *Entry) PrettyPrint(indent int) {
	fmt.Printf("%sDN: %s\n", strings.Repeat(" ", indent), e.DN)
	for _, attr := range e.Attributes {
		attr.PrettyPrint(indent + 2)
	}
}

type EntryAttribute struct {
	Name       string
	Values     []string
	ByteValues [][]byte
}

func (e *EntryAttribute) Print() {
	fmt.Printf("%s: %s\n", e.Name, e.Values)
}

func (e *EntryAttribute) PrettyPrint(indent int) {
	fmt.Printf("%s%s: %s\n", strings.Repeat(" ", indent), e.Name, e.Values)
}

type SearchResult struct {
	Entries   []*Entry
	Referrals []string
	Controls  []Control
}

func (s *SearchResult) Print() {
	for _, entry := range s.Entries {
		entry.Print()
	}
}

func (s *SearchResult) PrettyPrint(indent int) {
	for _, entry := range s.Entries {
		entry.PrettyPrint(indent)
	}
}

type SearchRequest struct {
	BaseDN       string
	Scope        int
	DerefAliases int
	SizeLimit    int
	TimeLimit    int
	TypesOnly    bool
	Filter       string
	Attributes   []string
	Controls     []Control
}

func (s *SearchRequest) encode() (*ber.Packet, error) {
	request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationSearchRequest, nil, "Search Request")
	request.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, s.BaseDN, "Base DN"))
	request.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagEnumerated, uint64(s.Scope), "Scope"))
	request.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagEnumerated, uint64(s.DerefAliases), "Deref Aliases"))
	request.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, uint64(s.SizeLimit), "Size Limit"))
	request.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, uint64(s.TimeLimit), "Time Limit"))
	request.AppendChild(ber.NewBoolean(ber.ClassUniversal, ber.TypePrimitive, ber.TagBoolean, s.TypesOnly, "Types Only"))
	// compile and encode filter
	filterPacket, err := CompileFilter(s.Filter)
	if err != nil {
		return nil, err
	}
	request.AppendChild(filterPacket)
	// encode attributes
	attributesPacket := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Attributes")
	for _, attribute := range s.Attributes {
		attributesPacket.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, attribute, "Attribute"))
	}
	request.AppendChild(attributesPacket)
	return request, nil
}

func NewSearchRequest(
	BaseDN string,
	Scope, DerefAliases, SizeLimit, TimeLimit int,
	TypesOnly bool,
	Filter string,
	Attributes []string,
	Controls []Control,
) *SearchRequest {
	return &SearchRequest{
		BaseDN:       BaseDN,
		Scope:        Scope,
		DerefAliases: DerefAliases,
		SizeLimit:    SizeLimit,
		TimeLimit:    TimeLimit,
		TypesOnly:    TypesOnly,
		Filter:       Filter,
		Attributes:   Attributes,
		Controls:     Controls,
	}
}

func (l *Conn) SearchWithPaging(searchRequest *SearchRequest, pagingSize uint32) (*SearchResult, error) {
	if searchRequest.Controls == nil {
		searchRequest.Controls = make([]Control, 0)
	}

	pagingControl := NewControlPaging(pagingSize)
	searchRequest.Controls = append(searchRequest.Controls, pagingControl)
	searchResult := new(SearchResult)
	for {
		result, err := l.Search(searchRequest)
		l.Debug.Printf("Looking for Paging Control...")
		if err != nil {
			return searchResult, err
		}
		if result == nil {
			return searchResult, NewError(ErrorNetwork, errors.New("ldap: packet not received"))
		}

		for _, entry := range result.Entries {
			searchResult.Entries = append(searchResult.Entries, entry)
		}
		for _, referral := range result.Referrals {
			searchResult.Referrals = append(searchResult.Referrals, referral)
		}
		for _, control := range result.Controls {
			searchResult.Controls = append(searchResult.Controls, control)
		}

		l.Debug.Printf("Looking for Paging Control...")
		pagingResult := FindControl(result.Controls, ControlTypePaging)
		if pagingResult == nil {
			pagingControl = nil
			l.Debug.Printf("Could not find paging control.  Breaking...")
			break
		}

		cookie := pagingResult.(*ControlPaging).Cookie
		if len(cookie) == 0 {
			pagingControl = nil
			l.Debug.Printf("Could not find cookie.  Breaking...")
			break
		}
		pagingControl.SetCookie(cookie)
	}

	if pagingControl != nil {
		l.Debug.Printf("Abandoning Paging...")
		pagingControl.PagingSize = 0
		l.Search(searchRequest)
	}

	return searchResult, nil
}

func (l *Conn) Search(searchRequest *SearchRequest) (*SearchResult, error) {
	messageID := l.nextMessageID()
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, messageID, "MessageID"))
	// encode search request
	encodedSearchRequest, err := searchRequest.encode()
	if err != nil {
		return nil, err
	}
	packet.AppendChild(encodedSearchRequest)
	// encode search controls
	if searchRequest.Controls != nil {
		packet.AppendChild(encodeControls(searchRequest.Controls))
	}

	l.Debug.PrintPacket(packet)

	channel, err := l.sendMessage(packet)
	if err != nil {
		return nil, err
	}
	if channel == nil {
		return nil, NewError(ErrorNetwork, errors.New("ldap: could not send message"))
	}
	defer l.finishMessage(messageID)

	result := &SearchResult{
		Entries:   make([]*Entry, 0),
		Referrals: make([]string, 0),
		Controls:  make([]Control, 0)}

	foundSearchResultDone := false
	for !foundSearchResultDone {
		l.Debug.Printf("%d: waiting for response", messageID)
		packet = <-channel
		l.Debug.Printf("%d: got response %p", messageID, packet)
		if packet == nil {
			return nil, NewError(ErrorNetwork, errors.New("ldap: could not retrieve message"))
		}

		if l.Debug {
			if err := addLDAPDescriptions(packet); err != nil {
				return nil, err
			}
			ber.PrintPacket(packet)
		}

		switch packet.Children[1].Tag {
		case 4:
			entry := new(Entry)
			entry.DN = packet.Children[1].Children[0].Value.(string)
			for _, child := range packet.Children[1].Children[1].Children {
				attr := new(EntryAttribute)
				attr.Name = child.Children[0].Value.(string)
				for _, value := range child.Children[1].Children {
					attr.Values = append(attr.Values, value.Value.(string))
					attr.ByteValues = append(attr.ByteValues, value.ByteValue)
				}
				entry.Attributes = append(entry.Attributes, attr)
			}
			result.Entries = append(result.Entries, entry)
		case 5:
			resultCode, resultDescription := getLDAPResultCode(packet)
			if resultCode != 0 {
				return result, NewError(resultCode, errors.New(resultDescription))
			}
			if len(packet.Children) == 3 {
				for _, child := range packet.Children[2].Children {
					result.Controls = append(result.Controls, DecodeControl(child))
				}
			}
			foundSearchResultDone = true
		case 19:
			result.Referrals = append(result.Referrals, packet.Children[1].Children[0].Value.(string))
		}
	}
	l.Debug.Printf("%d: returning", messageID)
	return result, nil
}
