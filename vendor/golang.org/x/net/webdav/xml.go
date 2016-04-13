// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webdav

// The XML encoding is covered by Section 14.
// http://www.webdav.org/specs/rfc4918.html#xml.element.definitions

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"time"

	"golang.org/x/net/webdav/internal/xml"
)

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_lockinfo
type lockInfo struct {
	XMLName   xml.Name  `xml:"lockinfo"`
	Exclusive *struct{} `xml:"lockscope>exclusive"`
	Shared    *struct{} `xml:"lockscope>shared"`
	Write     *struct{} `xml:"locktype>write"`
	Owner     owner     `xml:"owner"`
}

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_owner
type owner struct {
	InnerXML string `xml:",innerxml"`
}

func readLockInfo(r io.Reader) (li lockInfo, status int, err error) {
	c := &countingReader{r: r}
	if err = xml.NewDecoder(c).Decode(&li); err != nil {
		if err == io.EOF {
			if c.n == 0 {
				// An empty body means to refresh the lock.
				// http://www.webdav.org/specs/rfc4918.html#refreshing-locks
				return lockInfo{}, 0, nil
			}
			err = errInvalidLockInfo
		}
		return lockInfo{}, http.StatusBadRequest, err
	}
	// We only support exclusive (non-shared) write locks. In practice, these are
	// the only types of locks that seem to matter.
	if li.Exclusive == nil || li.Shared != nil || li.Write == nil {
		return lockInfo{}, http.StatusNotImplemented, errUnsupportedLockInfo
	}
	return li, 0, nil
}

type countingReader struct {
	n int
	r io.Reader
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += n
	return n, err
}

func writeLockInfo(w io.Writer, token string, ld LockDetails) (int, error) {
	depth := "infinity"
	if ld.ZeroDepth {
		depth = "0"
	}
	timeout := ld.Duration / time.Second
	return fmt.Fprintf(w, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"+
		"<D:prop xmlns:D=\"DAV:\"><D:lockdiscovery><D:activelock>\n"+
		"	<D:locktype><D:write/></D:locktype>\n"+
		"	<D:lockscope><D:exclusive/></D:lockscope>\n"+
		"	<D:depth>%s</D:depth>\n"+
		"	<D:owner>%s</D:owner>\n"+
		"	<D:timeout>Second-%d</D:timeout>\n"+
		"	<D:locktoken><D:href>%s</D:href></D:locktoken>\n"+
		"	<D:lockroot><D:href>%s</D:href></D:lockroot>\n"+
		"</D:activelock></D:lockdiscovery></D:prop>",
		depth, ld.OwnerXML, timeout, escape(token), escape(ld.Root),
	)
}

func escape(s string) string {
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"', '&', '\'', '<', '>':
			b := bytes.NewBuffer(nil)
			xml.EscapeText(b, []byte(s))
			return b.String()
		}
	}
	return s
}

// Next returns the next token, if any, in the XML stream of d.
// RFC 4918 requires to ignore comments, processing instructions
// and directives.
// http://www.webdav.org/specs/rfc4918.html#property_values
// http://www.webdav.org/specs/rfc4918.html#xml-extensibility
func next(d *xml.Decoder) (xml.Token, error) {
	for {
		t, err := d.Token()
		if err != nil {
			return t, err
		}
		switch t.(type) {
		case xml.Comment, xml.Directive, xml.ProcInst:
			continue
		default:
			return t, nil
		}
	}
}

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_prop (for propfind)
type propfindProps []xml.Name

// UnmarshalXML appends the property names enclosed within start to pn.
//
// It returns an error if start does not contain any properties or if
// properties contain values. Character data between properties is ignored.
func (pn *propfindProps) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	for {
		t, err := next(d)
		if err != nil {
			return err
		}
		switch t.(type) {
		case xml.EndElement:
			if len(*pn) == 0 {
				return fmt.Errorf("%s must not be empty", start.Name.Local)
			}
			return nil
		case xml.StartElement:
			name := t.(xml.StartElement).Name
			t, err = next(d)
			if err != nil {
				return err
			}
			if _, ok := t.(xml.EndElement); !ok {
				return fmt.Errorf("unexpected token %T", t)
			}
			*pn = append(*pn, name)
		}
	}
}

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_propfind
type propfind struct {
	XMLName  xml.Name      `xml:"DAV: propfind"`
	Allprop  *struct{}     `xml:"DAV: allprop"`
	Propname *struct{}     `xml:"DAV: propname"`
	Prop     propfindProps `xml:"DAV: prop"`
	Include  propfindProps `xml:"DAV: include"`
}

func readPropfind(r io.Reader) (pf propfind, status int, err error) {
	c := countingReader{r: r}
	if err = xml.NewDecoder(&c).Decode(&pf); err != nil {
		if err == io.EOF {
			if c.n == 0 {
				// An empty body means to propfind allprop.
				// http://www.webdav.org/specs/rfc4918.html#METHOD_PROPFIND
				return propfind{Allprop: new(struct{})}, 0, nil
			}
			err = errInvalidPropfind
		}
		return propfind{}, http.StatusBadRequest, err
	}

	if pf.Allprop == nil && pf.Include != nil {
		return propfind{}, http.StatusBadRequest, errInvalidPropfind
	}
	if pf.Allprop != nil && (pf.Prop != nil || pf.Propname != nil) {
		return propfind{}, http.StatusBadRequest, errInvalidPropfind
	}
	if pf.Prop != nil && pf.Propname != nil {
		return propfind{}, http.StatusBadRequest, errInvalidPropfind
	}
	if pf.Propname == nil && pf.Allprop == nil && pf.Prop == nil {
		return propfind{}, http.StatusBadRequest, errInvalidPropfind
	}
	return pf, 0, nil
}

// Property represents a single DAV resource property as defined in RFC 4918.
// See http://www.webdav.org/specs/rfc4918.html#data.model.for.resource.properties
type Property struct {
	// XMLName is the fully qualified name that identifies this property.
	XMLName xml.Name

	// Lang is an optional xml:lang attribute.
	Lang string `xml:"xml:lang,attr,omitempty"`

	// InnerXML contains the XML representation of the property value.
	// See http://www.webdav.org/specs/rfc4918.html#property_values
	//
	// Property values of complex type or mixed-content must have fully
	// expanded XML namespaces or be self-contained with according
	// XML namespace declarations. They must not rely on any XML
	// namespace declarations within the scope of the XML document,
	// even including the DAV: namespace.
	InnerXML []byte `xml:",innerxml"`
}

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_error
// See multistatusWriter for the "D:" namespace prefix.
type xmlError struct {
	XMLName  xml.Name `xml:"D:error"`
	InnerXML []byte   `xml:",innerxml"`
}

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_propstat
// See multistatusWriter for the "D:" namespace prefix.
type propstat struct {
	Prop                []Property `xml:"D:prop>_ignored_"`
	Status              string     `xml:"D:status"`
	Error               *xmlError  `xml:"D:error"`
	ResponseDescription string     `xml:"D:responsedescription,omitempty"`
}

// MarshalXML prepends the "D:" namespace prefix on properties in the DAV: namespace
// before encoding. See multistatusWriter.
func (ps propstat) MarshalXML(e *xml.Encoder, start xml.StartElement) error {
	for k, prop := range ps.Prop {
		if prop.XMLName.Space == "DAV:" {
			prop.XMLName = xml.Name{Space: "", Local: "D:" + prop.XMLName.Local}
			ps.Prop[k] = prop
		}
	}
	// Distinct type to avoid infinite recursion of MarshalXML.
	type newpropstat propstat
	return e.EncodeElement(newpropstat(ps), start)
}

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_response
// See multistatusWriter for the "D:" namespace prefix.
type response struct {
	XMLName             xml.Name   `xml:"D:response"`
	Href                []string   `xml:"D:href"`
	Propstat            []propstat `xml:"D:propstat"`
	Status              string     `xml:"D:status,omitempty"`
	Error               *xmlError  `xml:"D:error"`
	ResponseDescription string     `xml:"D:responsedescription,omitempty"`
}

// MultistatusWriter marshals one or more Responses into a XML
// multistatus response.
// See http://www.webdav.org/specs/rfc4918.html#ELEMENT_multistatus
// TODO(rsto, mpl): As a workaround, the "D:" namespace prefix, defined as
// "DAV:" on this element, is prepended on the nested response, as well as on all
// its nested elements. All property names in the DAV: namespace are prefixed as
// well. This is because some versions of Mini-Redirector (on windows 7) ignore
// elements with a default namespace (no prefixed namespace). A less intrusive fix
// should be possible after golang.org/cl/11074. See https://golang.org/issue/11177
type multistatusWriter struct {
	// ResponseDescription contains the optional responsedescription
	// of the multistatus XML element. Only the latest content before
	// close will be emitted. Empty response descriptions are not
	// written.
	responseDescription string

	w   http.ResponseWriter
	enc *xml.Encoder
}

// Write validates and emits a DAV response as part of a multistatus response
// element.
//
// It sets the HTTP status code of its underlying http.ResponseWriter to 207
// (Multi-Status) and populates the Content-Type header. If r is the
// first, valid response to be written, Write prepends the XML representation
// of r with a multistatus tag. Callers must call close after the last response
// has been written.
func (w *multistatusWriter) write(r *response) error {
	switch len(r.Href) {
	case 0:
		return errInvalidResponse
	case 1:
		if len(r.Propstat) > 0 != (r.Status == "") {
			return errInvalidResponse
		}
	default:
		if len(r.Propstat) > 0 || r.Status == "" {
			return errInvalidResponse
		}
	}
	err := w.writeHeader()
	if err != nil {
		return err
	}
	return w.enc.Encode(r)
}

// writeHeader writes a XML multistatus start element on w's underlying
// http.ResponseWriter and returns the result of the write operation.
// After the first write attempt, writeHeader becomes a no-op.
func (w *multistatusWriter) writeHeader() error {
	if w.enc != nil {
		return nil
	}
	w.w.Header().Add("Content-Type", "text/xml; charset=utf-8")
	w.w.WriteHeader(StatusMulti)
	_, err := fmt.Fprintf(w.w, `<?xml version="1.0" encoding="UTF-8"?>`)
	if err != nil {
		return err
	}
	w.enc = xml.NewEncoder(w.w)
	return w.enc.EncodeToken(xml.StartElement{
		Name: xml.Name{
			Space: "DAV:",
			Local: "multistatus",
		},
		Attr: []xml.Attr{{
			Name:  xml.Name{Space: "xmlns", Local: "D"},
			Value: "DAV:",
		}},
	})
}

// Close completes the marshalling of the multistatus response. It returns
// an error if the multistatus response could not be completed. If both the
// return value and field enc of w are nil, then no multistatus response has
// been written.
func (w *multistatusWriter) close() error {
	if w.enc == nil {
		return nil
	}
	var end []xml.Token
	if w.responseDescription != "" {
		name := xml.Name{Space: "DAV:", Local: "responsedescription"}
		end = append(end,
			xml.StartElement{Name: name},
			xml.CharData(w.responseDescription),
			xml.EndElement{Name: name},
		)
	}
	end = append(end, xml.EndElement{
		Name: xml.Name{Space: "DAV:", Local: "multistatus"},
	})
	for _, t := range end {
		err := w.enc.EncodeToken(t)
		if err != nil {
			return err
		}
	}
	return w.enc.Flush()
}

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_prop (for proppatch)
type proppatchProps []Property

var xmlLangName = xml.Name{Space: "http://www.w3.org/XML/1998/namespace", Local: "lang"}

func xmlLang(s xml.StartElement, d string) string {
	for _, attr := range s.Attr {
		if attr.Name == xmlLangName {
			return attr.Value
		}
	}
	return d
}

type xmlValue []byte

func (v *xmlValue) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	// The XML value of a property can be arbitrary, mixed-content XML.
	// To make sure that the unmarshalled value contains all required
	// namespaces, we encode all the property value XML tokens into a
	// buffer. This forces the encoder to redeclare any used namespaces.
	var b bytes.Buffer
	e := xml.NewEncoder(&b)
	for {
		t, err := next(d)
		if err != nil {
			return err
		}
		if e, ok := t.(xml.EndElement); ok && e.Name == start.Name {
			break
		}
		if err = e.EncodeToken(t); err != nil {
			return err
		}
	}
	err := e.Flush()
	if err != nil {
		return err
	}
	*v = b.Bytes()
	return nil
}

// UnmarshalXML appends the property names and values enclosed within start
// to ps.
//
// An xml:lang attribute that is defined either on the DAV:prop or property
// name XML element is propagated to the property's Lang field.
//
// UnmarshalXML returns an error if start does not contain any properties or if
// property values contain syntactically incorrect XML.
func (ps *proppatchProps) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	lang := xmlLang(start, "")
	for {
		t, err := next(d)
		if err != nil {
			return err
		}
		switch elem := t.(type) {
		case xml.EndElement:
			if len(*ps) == 0 {
				return fmt.Errorf("%s must not be empty", start.Name.Local)
			}
			return nil
		case xml.StartElement:
			p := Property{
				XMLName: t.(xml.StartElement).Name,
				Lang:    xmlLang(t.(xml.StartElement), lang),
			}
			err = d.DecodeElement(((*xmlValue)(&p.InnerXML)), &elem)
			if err != nil {
				return err
			}
			*ps = append(*ps, p)
		}
	}
}

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_set
// http://www.webdav.org/specs/rfc4918.html#ELEMENT_remove
type setRemove struct {
	XMLName xml.Name
	Lang    string         `xml:"xml:lang,attr,omitempty"`
	Prop    proppatchProps `xml:"DAV: prop"`
}

// http://www.webdav.org/specs/rfc4918.html#ELEMENT_propertyupdate
type propertyupdate struct {
	XMLName   xml.Name    `xml:"DAV: propertyupdate"`
	Lang      string      `xml:"xml:lang,attr,omitempty"`
	SetRemove []setRemove `xml:",any"`
}

func readProppatch(r io.Reader) (patches []Proppatch, status int, err error) {
	var pu propertyupdate
	if err = xml.NewDecoder(r).Decode(&pu); err != nil {
		return nil, http.StatusBadRequest, err
	}
	for _, op := range pu.SetRemove {
		remove := false
		switch op.XMLName {
		case xml.Name{Space: "DAV:", Local: "set"}:
			// No-op.
		case xml.Name{Space: "DAV:", Local: "remove"}:
			for _, p := range op.Prop {
				if len(p.InnerXML) > 0 {
					return nil, http.StatusBadRequest, errInvalidProppatch
				}
			}
			remove = true
		default:
			return nil, http.StatusBadRequest, errInvalidProppatch
		}
		patches = append(patches, Proppatch{Remove: remove, Props: op.Prop})
	}
	return patches, 0, nil
}
