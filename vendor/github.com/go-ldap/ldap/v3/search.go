package ldap

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	ber "github.com/go-asn1-ber/asn1-ber"
)

// scope choices
const (
	ScopeBaseObject   = 0
	ScopeSingleLevel  = 1
	ScopeWholeSubtree = 2
	// ScopeChildren is an OpenLDAP extension that may not be supported by another directory server.
	// See: https://github.com/openldap/openldap/blob/7c55484ee153047efd0e562fc1638c1a2525f320/include/ldap.h#L598
	ScopeChildren = 3
)

// ScopeMap contains human readable descriptions of scope choices
var ScopeMap = map[int]string{
	ScopeBaseObject:   "Base Object",
	ScopeSingleLevel:  "Single Level",
	ScopeWholeSubtree: "Whole Subtree",
	ScopeChildren:     "Children",
}

// derefAliases
const (
	NeverDerefAliases   = 0
	DerefInSearching    = 1
	DerefFindingBaseObj = 2
	DerefAlways         = 3
)

// DerefMap contains human readable descriptions of derefAliases choices
var DerefMap = map[int]string{
	NeverDerefAliases:   "NeverDerefAliases",
	DerefInSearching:    "DerefInSearching",
	DerefFindingBaseObj: "DerefFindingBaseObj",
	DerefAlways:         "DerefAlways",
}

// ErrSizeLimitExceeded will be returned if the search result is exceeding the defined SizeLimit
// and enforcing the requested limit is enabled in the search request (EnforceSizeLimit)
var ErrSizeLimitExceeded = NewError(ErrorNetwork, errors.New("ldap: size limit exceeded"))

// NewEntry returns an Entry object with the specified distinguished name and attribute key-value pairs.
// The map of attributes is accessed in alphabetical order of the keys in order to ensure that, for the
// same input map of attributes, the output entry will contain the same order of attributes
func NewEntry(dn string, attributes map[string][]string) *Entry {
	var attributeNames []string
	for attributeName := range attributes {
		attributeNames = append(attributeNames, attributeName)
	}
	sort.Strings(attributeNames)

	var encodedAttributes []*EntryAttribute
	for _, attributeName := range attributeNames {
		encodedAttributes = append(encodedAttributes, NewEntryAttribute(attributeName, attributes[attributeName]))
	}
	return &Entry{
		DN:         dn,
		Attributes: encodedAttributes,
	}
}

// Entry represents a single search result entry
type Entry struct {
	// DN is the distinguished name of the entry
	DN string
	// Attributes are the returned attributes for the entry
	Attributes []*EntryAttribute
}

// GetAttributeValues returns the values for the named attribute, or an empty list
func (e *Entry) GetAttributeValues(attribute string) []string {
	for _, attr := range e.Attributes {
		if attr.Name == attribute {
			return attr.Values
		}
	}
	return []string{}
}

// GetEqualFoldAttributeValues returns the values for the named attribute, or an
// empty list. Attribute matching is done with strings.EqualFold.
func (e *Entry) GetEqualFoldAttributeValues(attribute string) []string {
	for _, attr := range e.Attributes {
		if strings.EqualFold(attribute, attr.Name) {
			return attr.Values
		}
	}
	return []string{}
}

// GetRawAttributeValues returns the byte values for the named attribute, or an empty list
func (e *Entry) GetRawAttributeValues(attribute string) [][]byte {
	for _, attr := range e.Attributes {
		if attr.Name == attribute {
			return attr.ByteValues
		}
	}
	return [][]byte{}
}

// GetEqualFoldRawAttributeValues returns the byte values for the named attribute, or an empty list
func (e *Entry) GetEqualFoldRawAttributeValues(attribute string) [][]byte {
	for _, attr := range e.Attributes {
		if strings.EqualFold(attr.Name, attribute) {
			return attr.ByteValues
		}
	}
	return [][]byte{}
}

// GetAttributeValue returns the first value for the named attribute, or ""
func (e *Entry) GetAttributeValue(attribute string) string {
	values := e.GetAttributeValues(attribute)
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

// GetEqualFoldAttributeValue returns the first value for the named attribute, or "".
// Attribute comparison is done with strings.EqualFold.
func (e *Entry) GetEqualFoldAttributeValue(attribute string) string {
	values := e.GetEqualFoldAttributeValues(attribute)
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

// GetRawAttributeValue returns the first value for the named attribute, or an empty slice
func (e *Entry) GetRawAttributeValue(attribute string) []byte {
	values := e.GetRawAttributeValues(attribute)
	if len(values) == 0 {
		return []byte{}
	}
	return values[0]
}

// GetEqualFoldRawAttributeValue returns the first value for the named attribute, or an empty slice
func (e *Entry) GetEqualFoldRawAttributeValue(attribute string) []byte {
	values := e.GetEqualFoldRawAttributeValues(attribute)
	if len(values) == 0 {
		return []byte{}
	}
	return values[0]
}

// Print outputs a human-readable description
func (e *Entry) Print() {
	fmt.Printf("DN: %s\n", e.DN)
	for _, attr := range e.Attributes {
		attr.Print()
	}
}

// PrettyPrint outputs a human-readable description indenting
func (e *Entry) PrettyPrint(indent int) {
	fmt.Printf("%sDN: %s\n", strings.Repeat(" ", indent), e.DN)
	for _, attr := range e.Attributes {
		attr.PrettyPrint(indent + 2)
	}
}

// Describe the tag to use for struct field tags
const decoderTagName = "ldap"

// readTag will read the reflect.StructField value for
// the key defined in decoderTagName. If omitempty is
// specified, the field may not be filled.
func readTag(f reflect.StructField) (string, bool) {
	val, ok := f.Tag.Lookup(decoderTagName)
	if !ok {
		return f.Name, false
	}
	opts := strings.Split(val, ",")
	omit := false
	if len(opts) == 2 {
		omit = opts[1] == "omitempty"
	}
	return opts[0], omit
}

// Unmarshal parses the Entry in the value pointed to by i
//
// Currently, this methods only supports struct fields of type
// string, *string, []string, int, int64, []byte, *DN, []*DN or time.Time.
// Other field types will not be regarded. If the field type is a string or int but multiple
// attribute values are returned, the first value will be used to fill the field.
//
// Example:
//
//	type UserEntry struct {
//		// Fields with the tag key `dn` are automatically filled with the
//		// objects distinguishedName. This can be used multiple times.
//		DN string `ldap:"dn"`
//
//		// This field will be filled with the attribute value for
//		// userPrincipalName. An attribute can be read into a struct field
//		// multiple times. Missing attributes will not result in an error.
//		UserPrincipalName string `ldap:"userPrincipalName"`
//
//		// memberOf may have multiple values. If you don't
//		// know the amount of attribute values at runtime, use a string array.
//		MemberOf []string `ldap:"memberOf"`
//
//		// ID is an integer value, it will fail unmarshaling when the given
//		// attribute value cannot be parsed into an integer.
//		ID int `ldap:"id"`
//
//		// LongID is similar to ID but uses an int64 instead.
//		LongID int64 `ldap:"longId"`
//
//		// Data is similar to MemberOf a slice containing all attribute
//		// values.
//		Data []byte `ldap:"data"`
//
//		// Time is parsed with the generalizedTime spec into a time.Time
//		Created time.Time `ldap:"createdTimestamp"`
//
//		// *DN is parsed with the ParseDN
//		Owner *ldap.DN `ldap:"owner"`
//
//		// []*DN is parsed with the ParseDN
//		Children []*ldap.DN `ldap:"children"`
//
//		// This won't work, as the field is not of type string. For this
//		// to work, you'll have to temporarily store the result in string
//		// (or string array) and convert it to the desired type afterwards.
//		UserAccountControl uint32 `ldap:"userPrincipalName"`
//	}
//	user := UserEntry{}
//
//	if err := result.Unmarshal(&user); err != nil {
//		// ...
//	}
func (e *Entry) Unmarshal(i interface{}) (err error) {
	// Make sure it's a ptr
	if vo := reflect.ValueOf(i).Kind(); vo != reflect.Ptr {
		return fmt.Errorf("ldap: cannot use %s, expected pointer to a struct", vo)
	}

	sv, st := reflect.ValueOf(i).Elem(), reflect.TypeOf(i).Elem()
	// Make sure it's pointing to a struct
	if sv.Kind() != reflect.Struct {
		return fmt.Errorf("ldap: expected pointer to a struct, got %s", sv.Kind())
	}

	for n := 0; n < st.NumField(); n++ {
		// Holds struct field value and type
		fv, ft := sv.Field(n), st.Field(n)

		// skip unexported fields
		if ft.PkgPath != "" {
			continue
		}

		// omitempty can be safely discarded, as it's not needed when unmarshalling
		fieldTag, _ := readTag(ft)

		// Fill the field with the distinguishedName if the tag key is `dn`
		if fieldTag == "dn" {
			fv.SetString(e.DN)
			continue
		}

		values := e.GetAttributeValues(fieldTag)
		if len(values) == 0 {
			continue
		}

		switch fv.Interface().(type) {
		case []string:
			for _, item := range values {
				fv.Set(reflect.Append(fv, reflect.ValueOf(item)))
			}
		case string:
			fv.SetString(values[0])
		case *string:
			fv.Set(reflect.ValueOf(&values[0]))
		case []byte:
			fv.SetBytes([]byte(values[0]))
		case int, int64:
			intVal, err := strconv.ParseInt(values[0], 10, 64)
			if err != nil {
				return fmt.Errorf("ldap: could not parse value '%s' into int field", values[0])
			}
			fv.SetInt(intVal)
		case time.Time:
			t, err := ber.ParseGeneralizedTime([]byte(values[0]))
			if err != nil {
				return fmt.Errorf("ldap: could not parse value '%s' into time.Time field", values[0])
			}
			fv.Set(reflect.ValueOf(t))
		case *DN:
			dn, err := ParseDN(values[0])
			if err != nil {
				return fmt.Errorf("ldap: could not parse value '%s' into *ldap.DN field", values[0])
			}
			fv.Set(reflect.ValueOf(dn))
		case []*DN:
			for _, item := range values {
				dn, err := ParseDN(item)
				if err != nil {
					return fmt.Errorf("ldap: could not parse value '%s' into *ldap.DN field", item)
				}
				fv.Set(reflect.Append(fv, reflect.ValueOf(dn)))
			}
		default:
			return fmt.Errorf("ldap: expected field to be of type string, *string, []string, int, int64, []byte, *DN, []*DN or time.Time, got %v", ft.Type)
		}
	}
	return
}

// NewEntryAttribute returns a new EntryAttribute with the desired key-value pair
func NewEntryAttribute(name string, values []string) *EntryAttribute {
	var bytes [][]byte
	for _, value := range values {
		bytes = append(bytes, []byte(value))
	}
	return &EntryAttribute{
		Name:       name,
		Values:     values,
		ByteValues: bytes,
	}
}

// EntryAttribute holds a single attribute
type EntryAttribute struct {
	// Name is the name of the attribute
	Name string
	// Values contain the string values of the attribute
	Values []string
	// ByteValues contain the raw values of the attribute
	ByteValues [][]byte
}

// Print outputs a human-readable description
func (e *EntryAttribute) Print() {
	fmt.Printf("%s: %s\n", e.Name, e.Values)
}

// PrettyPrint outputs a human-readable description with indenting
func (e *EntryAttribute) PrettyPrint(indent int) {
	fmt.Printf("%s%s: %s\n", strings.Repeat(" ", indent), e.Name, e.Values)
}

// SearchResult holds the server's response to a search request
type SearchResult struct {
	// Entries are the returned entries
	Entries []*Entry
	// Referrals are the returned referrals
	Referrals []string
	// Controls are the returned controls
	Controls []Control
}

// Print outputs a human-readable description
func (s *SearchResult) Print() {
	for _, entry := range s.Entries {
		entry.Print()
	}
}

// PrettyPrint outputs a human-readable description with indenting
func (s *SearchResult) PrettyPrint(indent int) {
	for _, entry := range s.Entries {
		entry.PrettyPrint(indent)
	}
}

// appendTo appends all entries of `s` to `r`
func (s *SearchResult) appendTo(r *SearchResult) {
	r.Entries = append(r.Entries, s.Entries...)
	r.Referrals = append(r.Referrals, s.Referrals...)
	r.Controls = append(r.Controls, s.Controls...)
}

// SearchSingleResult holds the server's single entry response to a search request
type SearchSingleResult struct {
	// Entry is the returned entry
	Entry *Entry
	// Referral is the returned referral
	Referral string
	// Controls are the returned controls
	Controls []Control
	// Error is set when the search request was failed
	Error error
}

// Print outputs a human-readable description
func (s *SearchSingleResult) Print() {
	s.Entry.Print()
}

// PrettyPrint outputs a human-readable description with indenting
func (s *SearchSingleResult) PrettyPrint(indent int) {
	s.Entry.PrettyPrint(indent)
}

// SearchRequest represents a search request to send to the server
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

	// EnforceSizeLimit will hard limit the maximum number of entries parsed, in case the directory
	// server returns more results than requested. This setting is disabled by default and does not
	// work in async search requests.
	EnforceSizeLimit bool
}

func (req *SearchRequest) appendTo(envelope *ber.Packet) error {
	pkt := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationSearchRequest, nil, "Search Request")
	pkt.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, req.BaseDN, "Base DN"))
	pkt.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagEnumerated, uint64(req.Scope), "Scope"))
	pkt.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagEnumerated, uint64(req.DerefAliases), "Deref Aliases"))
	pkt.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, uint64(req.SizeLimit), "Size Limit"))
	pkt.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, uint64(req.TimeLimit), "Time Limit"))
	pkt.AppendChild(ber.NewBoolean(ber.ClassUniversal, ber.TypePrimitive, ber.TagBoolean, req.TypesOnly, "Types Only"))
	// compile and encode filter
	filterPacket, err := CompileFilter(req.Filter)
	if err != nil {
		return err
	}
	pkt.AppendChild(filterPacket)
	// encode attributes
	attributesPacket := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Attributes")
	for _, attribute := range req.Attributes {
		attributesPacket.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, attribute, "Attribute"))
	}
	pkt.AppendChild(attributesPacket)

	envelope.AppendChild(pkt)
	if len(req.Controls) > 0 {
		envelope.AppendChild(encodeControls(req.Controls))
	}

	return nil
}

// NewSearchRequest creates a new search request
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

// SearchWithPaging accepts a search request and desired page size in order to execute LDAP queries to fulfill the
// search request. All paged LDAP query responses will be buffered and the final result will be returned atomically.
// The following four cases are possible given the arguments:
//   - given SearchRequest missing a control of type ControlTypePaging: we will add one with the desired paging size
//   - given SearchRequest contains a control of type ControlTypePaging that isn't actually a ControlPaging: fail without issuing any queries
//   - given SearchRequest contains a control of type ControlTypePaging with pagingSize equal to the size requested: no change to the search request
//   - given SearchRequest contains a control of type ControlTypePaging with pagingSize not equal to the size requested: fail without issuing any queries
//
// A requested pagingSize of 0 is interpreted as no limit by LDAP servers.
func (l *Conn) SearchWithPaging(searchRequest *SearchRequest, pagingSize uint32) (*SearchResult, error) {
	var pagingControl *ControlPaging

	control := FindControl(searchRequest.Controls, ControlTypePaging)
	if control == nil {
		pagingControl = NewControlPaging(pagingSize)
		searchRequest.Controls = append(searchRequest.Controls, pagingControl)
	} else {
		castControl, ok := control.(*ControlPaging)
		if !ok {
			return nil, fmt.Errorf("expected paging control to be of type *ControlPaging, got %v", control)
		}
		if castControl.PagingSize != pagingSize {
			return nil, fmt.Errorf("paging size given in search request (%d) conflicts with size given in search call (%d)", castControl.PagingSize, pagingSize)
		}
		pagingControl = castControl
	}

	searchResult := new(SearchResult)
	for {
		result, err := l.Search(searchRequest)
		if result != nil {
			result.appendTo(searchResult)
		} else {
			if err == nil {
				// We have to do this beautifulness in case something absolutely strange happens, which
				// should only occur in case there is no packet, but also no error.
				return searchResult, NewError(ErrorNetwork, errors.New("ldap: packet not received"))
			}
		}
		if err != nil {
			// If an error occurred, all results that have been received so far will be returned
			return searchResult, err
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
		if _, err := l.Search(searchRequest); err != nil {
			return searchResult, err
		}
	}

	return searchResult, nil
}

// Search performs the given search request
func (l *Conn) Search(searchRequest *SearchRequest) (*SearchResult, error) {
	msgCtx, err := l.doRequest(searchRequest)
	if err != nil {
		return nil, err
	}
	defer l.finishMessage(msgCtx)

	result := &SearchResult{
		Entries:   make([]*Entry, 0),
		Referrals: make([]string, 0),
		Controls:  make([]Control, 0),
	}

	for {
		packet, err := l.readPacket(msgCtx)
		if err != nil {
			return result, err
		}

		switch packet.Children[1].Tag {
		case 4:
			if searchRequest.EnforceSizeLimit &&
				searchRequest.SizeLimit > 0 &&
				len(result.Entries) >= searchRequest.SizeLimit {
				return result, ErrSizeLimitExceeded
			}

			attr := make([]*ber.Packet, 0)
			if len(packet.Children[1].Children) > 1 {
				attr = packet.Children[1].Children[1].Children
			}
			entry := &Entry{
				DN:         packet.Children[1].Children[0].Value.(string),
				Attributes: unpackAttributes(attr),
			}
			result.Entries = append(result.Entries, entry)
		case 5:
			err := GetLDAPError(packet)
			if err != nil {
				return result, err
			}
			if len(packet.Children) == 3 {
				for _, child := range packet.Children[2].Children {
					decodedChild, err := DecodeControl(child)
					if err != nil {
						return result, fmt.Errorf("failed to decode child control: %s", err)
					}
					result.Controls = append(result.Controls, decodedChild)
				}
			}
			return result, nil
		case 19:
			result.Referrals = append(result.Referrals, packet.Children[1].Children[0].Value.(string))
		}
	}
}

// SearchAsync performs a search request and returns all search results asynchronously.
// This means you get all results until an error happens (or the search successfully finished),
// e.g. for size / time limited requests all are recieved until the limit is reached.
// To stop the search, call cancel function of the context.
func (l *Conn) SearchAsync(
	ctx context.Context, searchRequest *SearchRequest, bufferSize int) Response {
	r := newSearchResponse(l, bufferSize)
	r.start(ctx, searchRequest)
	return r
}

// Syncrepl is a short name for LDAP Sync Replication engine that works on the
// consumer-side. This can perform a persistent search and returns an entry
// when the entry is updated on the server side.
// To stop the search, call cancel function of the context.
func (l *Conn) Syncrepl(
	ctx context.Context, searchRequest *SearchRequest, bufferSize int,
	mode ControlSyncRequestMode, cookie []byte, reloadHint bool,
) Response {
	control := NewControlSyncRequest(mode, cookie, reloadHint)
	searchRequest.Controls = append(searchRequest.Controls, control)
	r := newSearchResponse(l, bufferSize)
	r.start(ctx, searchRequest)
	return r
}

// unpackAttributes will extract all given LDAP attributes and it's values
// from the ber.Packet
func unpackAttributes(children []*ber.Packet) []*EntryAttribute {
	entries := make([]*EntryAttribute, len(children))
	for i, child := range children {
		length := len(child.Children[1].Children)
		entry := &EntryAttribute{
			Name: child.Children[0].Value.(string),
			// pre-allocate the slice since we can determine
			// the number of attributes at this point
			Values:     make([]string, length),
			ByteValues: make([][]byte, length),
		}

		for i, value := range child.Children[1].Children {
			entry.ByteValues[i] = value.ByteValue
			entry.Values[i] = value.Value.(string)
		}
		entries[i] = entry
	}

	return entries
}

// DirSync does a Search with dirSync Control.
func (l *Conn) DirSync(
	searchRequest *SearchRequest, flags int64, maxAttrCount int64, cookie []byte,
) (*SearchResult, error) {
	control := FindControl(searchRequest.Controls, ControlTypeDirSync)
	if control == nil {
		c := NewRequestControlDirSync(flags, maxAttrCount, cookie)
		searchRequest.Controls = append(searchRequest.Controls, c)
	} else {
		c := control.(*ControlDirSync)
		if c.Flags != flags {
			return nil, fmt.Errorf("flags given in search request (%d) conflicts with flags given in search call (%d)", c.Flags, flags)
		}
		if c.MaxAttrCount != maxAttrCount {
			return nil, fmt.Errorf("MaxAttrCnt given in search request (%d) conflicts with maxAttrCount given in search call (%d)", c.MaxAttrCount, maxAttrCount)
		}
	}
	searchResult, err := l.Search(searchRequest)
	l.Debug.Printf("Looking for result...")
	if err != nil {
		return nil, err
	}
	if searchResult == nil {
		return nil, NewError(ErrorNetwork, errors.New("ldap: packet not received"))
	}

	l.Debug.Printf("Looking for DirSync Control...")
	resultControl := FindControl(searchResult.Controls, ControlTypeDirSync)
	if resultControl == nil {
		l.Debug.Printf("Could not find dirSyncControl control.  Breaking...")
		return searchResult, nil
	}

	cookie = resultControl.(*ControlDirSync).Cookie
	if len(cookie) == 0 {
		l.Debug.Printf("Could not find cookie.  Breaking...")
		return searchResult, nil
	}

	return searchResult, nil
}

// DirSyncDirSyncAsync performs a search request and returns all search results
// asynchronously. This is efficient when the server returns lots of entries.
func (l *Conn) DirSyncAsync(
	ctx context.Context, searchRequest *SearchRequest, bufferSize int,
	flags, maxAttrCount int64, cookie []byte,
) Response {
	control := NewRequestControlDirSync(flags, maxAttrCount, cookie)
	searchRequest.Controls = append(searchRequest.Controls, control)
	r := newSearchResponse(l, bufferSize)
	r.start(ctx, searchRequest)
	return r
}
