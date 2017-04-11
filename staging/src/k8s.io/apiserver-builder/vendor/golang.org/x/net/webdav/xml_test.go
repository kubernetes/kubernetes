// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webdav

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"testing"

	ixml "golang.org/x/net/webdav/internal/xml"
)

func TestReadLockInfo(t *testing.T) {
	// The "section x.y.z" test cases come from section x.y.z of the spec at
	// http://www.webdav.org/specs/rfc4918.html
	testCases := []struct {
		desc       string
		input      string
		wantLI     lockInfo
		wantStatus int
	}{{
		"bad: junk",
		"xxx",
		lockInfo{},
		http.StatusBadRequest,
	}, {
		"bad: invalid owner XML",
		"" +
			"<D:lockinfo xmlns:D='DAV:'>\n" +
			"  <D:lockscope><D:exclusive/></D:lockscope>\n" +
			"  <D:locktype><D:write/></D:locktype>\n" +
			"  <D:owner>\n" +
			"    <D:href>   no end tag   \n" +
			"  </D:owner>\n" +
			"</D:lockinfo>",
		lockInfo{},
		http.StatusBadRequest,
	}, {
		"bad: invalid UTF-8",
		"" +
			"<D:lockinfo xmlns:D='DAV:'>\n" +
			"  <D:lockscope><D:exclusive/></D:lockscope>\n" +
			"  <D:locktype><D:write/></D:locktype>\n" +
			"  <D:owner>\n" +
			"    <D:href>   \xff   </D:href>\n" +
			"  </D:owner>\n" +
			"</D:lockinfo>",
		lockInfo{},
		http.StatusBadRequest,
	}, {
		"bad: unfinished XML #1",
		"" +
			"<D:lockinfo xmlns:D='DAV:'>\n" +
			"  <D:lockscope><D:exclusive/></D:lockscope>\n" +
			"  <D:locktype><D:write/></D:locktype>\n",
		lockInfo{},
		http.StatusBadRequest,
	}, {
		"bad: unfinished XML #2",
		"" +
			"<D:lockinfo xmlns:D='DAV:'>\n" +
			"  <D:lockscope><D:exclusive/></D:lockscope>\n" +
			"  <D:locktype><D:write/></D:locktype>\n" +
			"  <D:owner>\n",
		lockInfo{},
		http.StatusBadRequest,
	}, {
		"good: empty",
		"",
		lockInfo{},
		0,
	}, {
		"good: plain-text owner",
		"" +
			"<D:lockinfo xmlns:D='DAV:'>\n" +
			"  <D:lockscope><D:exclusive/></D:lockscope>\n" +
			"  <D:locktype><D:write/></D:locktype>\n" +
			"  <D:owner>gopher</D:owner>\n" +
			"</D:lockinfo>",
		lockInfo{
			XMLName:   ixml.Name{Space: "DAV:", Local: "lockinfo"},
			Exclusive: new(struct{}),
			Write:     new(struct{}),
			Owner: owner{
				InnerXML: "gopher",
			},
		},
		0,
	}, {
		"section 9.10.7",
		"" +
			"<D:lockinfo xmlns:D='DAV:'>\n" +
			"  <D:lockscope><D:exclusive/></D:lockscope>\n" +
			"  <D:locktype><D:write/></D:locktype>\n" +
			"  <D:owner>\n" +
			"    <D:href>http://example.org/~ejw/contact.html</D:href>\n" +
			"  </D:owner>\n" +
			"</D:lockinfo>",
		lockInfo{
			XMLName:   ixml.Name{Space: "DAV:", Local: "lockinfo"},
			Exclusive: new(struct{}),
			Write:     new(struct{}),
			Owner: owner{
				InnerXML: "\n    <D:href>http://example.org/~ejw/contact.html</D:href>\n  ",
			},
		},
		0,
	}}

	for _, tc := range testCases {
		li, status, err := readLockInfo(strings.NewReader(tc.input))
		if tc.wantStatus != 0 {
			if err == nil {
				t.Errorf("%s: got nil error, want non-nil", tc.desc)
				continue
			}
		} else if err != nil {
			t.Errorf("%s: %v", tc.desc, err)
			continue
		}
		if !reflect.DeepEqual(li, tc.wantLI) || status != tc.wantStatus {
			t.Errorf("%s:\ngot  lockInfo=%v, status=%v\nwant lockInfo=%v, status=%v",
				tc.desc, li, status, tc.wantLI, tc.wantStatus)
			continue
		}
	}
}

func TestReadPropfind(t *testing.T) {
	testCases := []struct {
		desc       string
		input      string
		wantPF     propfind
		wantStatus int
	}{{
		desc: "propfind: propname",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:propname/>\n" +
			"</A:propfind>",
		wantPF: propfind{
			XMLName:  ixml.Name{Space: "DAV:", Local: "propfind"},
			Propname: new(struct{}),
		},
	}, {
		desc:  "propfind: empty body means allprop",
		input: "",
		wantPF: propfind{
			Allprop: new(struct{}),
		},
	}, {
		desc: "propfind: allprop",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"   <A:allprop/>\n" +
			"</A:propfind>",
		wantPF: propfind{
			XMLName: ixml.Name{Space: "DAV:", Local: "propfind"},
			Allprop: new(struct{}),
		},
	}, {
		desc: "propfind: allprop followed by include",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:allprop/>\n" +
			"  <A:include><A:displayname/></A:include>\n" +
			"</A:propfind>",
		wantPF: propfind{
			XMLName: ixml.Name{Space: "DAV:", Local: "propfind"},
			Allprop: new(struct{}),
			Include: propfindProps{xml.Name{Space: "DAV:", Local: "displayname"}},
		},
	}, {
		desc: "propfind: include followed by allprop",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:include><A:displayname/></A:include>\n" +
			"  <A:allprop/>\n" +
			"</A:propfind>",
		wantPF: propfind{
			XMLName: ixml.Name{Space: "DAV:", Local: "propfind"},
			Allprop: new(struct{}),
			Include: propfindProps{xml.Name{Space: "DAV:", Local: "displayname"}},
		},
	}, {
		desc: "propfind: propfind",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop><A:displayname/></A:prop>\n" +
			"</A:propfind>",
		wantPF: propfind{
			XMLName: ixml.Name{Space: "DAV:", Local: "propfind"},
			Prop:    propfindProps{xml.Name{Space: "DAV:", Local: "displayname"}},
		},
	}, {
		desc: "propfind: prop with ignored comments",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop>\n" +
			"    <!-- ignore -->\n" +
			"    <A:displayname><!-- ignore --></A:displayname>\n" +
			"  </A:prop>\n" +
			"</A:propfind>",
		wantPF: propfind{
			XMLName: ixml.Name{Space: "DAV:", Local: "propfind"},
			Prop:    propfindProps{xml.Name{Space: "DAV:", Local: "displayname"}},
		},
	}, {
		desc: "propfind: propfind with ignored whitespace",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop>   <A:displayname/></A:prop>\n" +
			"</A:propfind>",
		wantPF: propfind{
			XMLName: ixml.Name{Space: "DAV:", Local: "propfind"},
			Prop:    propfindProps{xml.Name{Space: "DAV:", Local: "displayname"}},
		},
	}, {
		desc: "propfind: propfind with ignored mixed-content",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop>foo<A:displayname/>bar</A:prop>\n" +
			"</A:propfind>",
		wantPF: propfind{
			XMLName: ixml.Name{Space: "DAV:", Local: "propfind"},
			Prop:    propfindProps{xml.Name{Space: "DAV:", Local: "displayname"}},
		},
	}, {
		desc: "propfind: propname with ignored element (section A.4)",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:propname/>\n" +
			"  <E:leave-out xmlns:E='E:'>*boss*</E:leave-out>\n" +
			"</A:propfind>",
		wantPF: propfind{
			XMLName:  ixml.Name{Space: "DAV:", Local: "propfind"},
			Propname: new(struct{}),
		},
	}, {
		desc:       "propfind: bad: junk",
		input:      "xxx",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "propfind: bad: propname and allprop (section A.3)",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:propname/>" +
			"  <A:allprop/>" +
			"</A:propfind>",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "propfind: bad: propname and prop",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop><A:displayname/></A:prop>\n" +
			"  <A:propname/>\n" +
			"</A:propfind>",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "propfind: bad: allprop and prop",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:allprop/>\n" +
			"  <A:prop><A:foo/><A:/prop>\n" +
			"</A:propfind>",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "propfind: bad: empty propfind with ignored element (section A.4)",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <E:expired-props/>\n" +
			"</A:propfind>",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "propfind: bad: empty prop",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop/>\n" +
			"</A:propfind>",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "propfind: bad: prop with just chardata",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop>foo</A:prop>\n" +
			"</A:propfind>",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "bad: interrupted prop",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop><A:foo></A:prop>\n",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "bad: malformed end element prop",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop><A:foo/></A:bar></A:prop>\n",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "propfind: bad: property with chardata value",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop><A:foo>bar</A:foo></A:prop>\n" +
			"</A:propfind>",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "propfind: bad: property with whitespace value",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:prop><A:foo> </A:foo></A:prop>\n" +
			"</A:propfind>",
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "propfind: bad: include without allprop",
		input: "" +
			"<A:propfind xmlns:A='DAV:'>\n" +
			"  <A:include><A:foo/></A:include>\n" +
			"</A:propfind>",
		wantStatus: http.StatusBadRequest,
	}}

	for _, tc := range testCases {
		pf, status, err := readPropfind(strings.NewReader(tc.input))
		if tc.wantStatus != 0 {
			if err == nil {
				t.Errorf("%s: got nil error, want non-nil", tc.desc)
				continue
			}
		} else if err != nil {
			t.Errorf("%s: %v", tc.desc, err)
			continue
		}
		if !reflect.DeepEqual(pf, tc.wantPF) || status != tc.wantStatus {
			t.Errorf("%s:\ngot  propfind=%v, status=%v\nwant propfind=%v, status=%v",
				tc.desc, pf, status, tc.wantPF, tc.wantStatus)
			continue
		}
	}
}

func TestMultistatusWriter(t *testing.T) {
	///The "section x.y.z" test cases come from section x.y.z of the spec at
	// http://www.webdav.org/specs/rfc4918.html
	testCases := []struct {
		desc        string
		responses   []response
		respdesc    string
		writeHeader bool
		wantXML     string
		wantCode    int
		wantErr     error
	}{{
		desc: "section 9.2.2 (failed dependency)",
		responses: []response{{
			Href: []string{"http://example.com/foo"},
			Propstat: []propstat{{
				Prop: []Property{{
					XMLName: xml.Name{
						Space: "http://ns.example.com/",
						Local: "Authors",
					},
				}},
				Status: "HTTP/1.1 424 Failed Dependency",
			}, {
				Prop: []Property{{
					XMLName: xml.Name{
						Space: "http://ns.example.com/",
						Local: "Copyright-Owner",
					},
				}},
				Status: "HTTP/1.1 409 Conflict",
			}},
			ResponseDescription: "Copyright Owner cannot be deleted or altered.",
		}},
		wantXML: `` +
			`<?xml version="1.0" encoding="UTF-8"?>` +
			`<multistatus xmlns="DAV:">` +
			`  <response>` +
			`    <href>http://example.com/foo</href>` +
			`    <propstat>` +
			`      <prop>` +
			`        <Authors xmlns="http://ns.example.com/"></Authors>` +
			`      </prop>` +
			`      <status>HTTP/1.1 424 Failed Dependency</status>` +
			`    </propstat>` +
			`    <propstat xmlns="DAV:">` +
			`      <prop>` +
			`        <Copyright-Owner xmlns="http://ns.example.com/"></Copyright-Owner>` +
			`      </prop>` +
			`      <status>HTTP/1.1 409 Conflict</status>` +
			`    </propstat>` +
			`  <responsedescription>Copyright Owner cannot be deleted or altered.</responsedescription>` +
			`</response>` +
			`</multistatus>`,
		wantCode: StatusMulti,
	}, {
		desc: "section 9.6.2 (lock-token-submitted)",
		responses: []response{{
			Href:   []string{"http://example.com/foo"},
			Status: "HTTP/1.1 423 Locked",
			Error: &xmlError{
				InnerXML: []byte(`<lock-token-submitted xmlns="DAV:"/>`),
			},
		}},
		wantXML: `` +
			`<?xml version="1.0" encoding="UTF-8"?>` +
			`<multistatus xmlns="DAV:">` +
			`  <response>` +
			`    <href>http://example.com/foo</href>` +
			`    <status>HTTP/1.1 423 Locked</status>` +
			`    <error><lock-token-submitted xmlns="DAV:"/></error>` +
			`  </response>` +
			`</multistatus>`,
		wantCode: StatusMulti,
	}, {
		desc: "section 9.1.3",
		responses: []response{{
			Href: []string{"http://example.com/foo"},
			Propstat: []propstat{{
				Prop: []Property{{
					XMLName: xml.Name{Space: "http://ns.example.com/boxschema/", Local: "bigbox"},
					InnerXML: []byte(`` +
						`<BoxType xmlns="http://ns.example.com/boxschema/">` +
						`Box type A` +
						`</BoxType>`),
				}, {
					XMLName: xml.Name{Space: "http://ns.example.com/boxschema/", Local: "author"},
					InnerXML: []byte(`` +
						`<Name xmlns="http://ns.example.com/boxschema/">` +
						`J.J. Johnson` +
						`</Name>`),
				}},
				Status: "HTTP/1.1 200 OK",
			}, {
				Prop: []Property{{
					XMLName: xml.Name{Space: "http://ns.example.com/boxschema/", Local: "DingALing"},
				}, {
					XMLName: xml.Name{Space: "http://ns.example.com/boxschema/", Local: "Random"},
				}},
				Status:              "HTTP/1.1 403 Forbidden",
				ResponseDescription: "The user does not have access to the DingALing property.",
			}},
		}},
		respdesc: "There has been an access violation error.",
		wantXML: `` +
			`<?xml version="1.0" encoding="UTF-8"?>` +
			`<multistatus xmlns="DAV:" xmlns:B="http://ns.example.com/boxschema/">` +
			`  <response>` +
			`    <href>http://example.com/foo</href>` +
			`    <propstat>` +
			`      <prop>` +
			`        <B:bigbox><B:BoxType>Box type A</B:BoxType></B:bigbox>` +
			`        <B:author><B:Name>J.J. Johnson</B:Name></B:author>` +
			`      </prop>` +
			`      <status>HTTP/1.1 200 OK</status>` +
			`    </propstat>` +
			`    <propstat>` +
			`      <prop>` +
			`        <B:DingALing/>` +
			`        <B:Random/>` +
			`      </prop>` +
			`      <status>HTTP/1.1 403 Forbidden</status>` +
			`      <responsedescription>The user does not have access to the DingALing property.</responsedescription>` +
			`    </propstat>` +
			`  </response>` +
			`  <responsedescription>There has been an access violation error.</responsedescription>` +
			`</multistatus>`,
		wantCode: StatusMulti,
	}, {
		desc: "no response written",
		// default of http.responseWriter
		wantCode: http.StatusOK,
	}, {
		desc:     "no response written (with description)",
		respdesc: "too bad",
		// default of http.responseWriter
		wantCode: http.StatusOK,
	}, {
		desc:        "empty multistatus with header",
		writeHeader: true,
		wantXML:     `<multistatus xmlns="DAV:"></multistatus>`,
		wantCode:    StatusMulti,
	}, {
		desc: "bad: no href",
		responses: []response{{
			Propstat: []propstat{{
				Prop: []Property{{
					XMLName: xml.Name{
						Space: "http://example.com/",
						Local: "foo",
					},
				}},
				Status: "HTTP/1.1 200 OK",
			}},
		}},
		wantErr: errInvalidResponse,
		// default of http.responseWriter
		wantCode: http.StatusOK,
	}, {
		desc: "bad: multiple hrefs and no status",
		responses: []response{{
			Href: []string{"http://example.com/foo", "http://example.com/bar"},
		}},
		wantErr: errInvalidResponse,
		// default of http.responseWriter
		wantCode: http.StatusOK,
	}, {
		desc: "bad: one href and no propstat",
		responses: []response{{
			Href: []string{"http://example.com/foo"},
		}},
		wantErr: errInvalidResponse,
		// default of http.responseWriter
		wantCode: http.StatusOK,
	}, {
		desc: "bad: status with one href and propstat",
		responses: []response{{
			Href: []string{"http://example.com/foo"},
			Propstat: []propstat{{
				Prop: []Property{{
					XMLName: xml.Name{
						Space: "http://example.com/",
						Local: "foo",
					},
				}},
				Status: "HTTP/1.1 200 OK",
			}},
			Status: "HTTP/1.1 200 OK",
		}},
		wantErr: errInvalidResponse,
		// default of http.responseWriter
		wantCode: http.StatusOK,
	}, {
		desc: "bad: multiple hrefs and propstat",
		responses: []response{{
			Href: []string{
				"http://example.com/foo",
				"http://example.com/bar",
			},
			Propstat: []propstat{{
				Prop: []Property{{
					XMLName: xml.Name{
						Space: "http://example.com/",
						Local: "foo",
					},
				}},
				Status: "HTTP/1.1 200 OK",
			}},
		}},
		wantErr: errInvalidResponse,
		// default of http.responseWriter
		wantCode: http.StatusOK,
	}}

	n := xmlNormalizer{omitWhitespace: true}
loop:
	for _, tc := range testCases {
		rec := httptest.NewRecorder()
		w := multistatusWriter{w: rec, responseDescription: tc.respdesc}
		if tc.writeHeader {
			if err := w.writeHeader(); err != nil {
				t.Errorf("%s: got writeHeader error %v, want nil", tc.desc, err)
				continue
			}
		}
		for _, r := range tc.responses {
			if err := w.write(&r); err != nil {
				if err != tc.wantErr {
					t.Errorf("%s: got write error %v, want %v",
						tc.desc, err, tc.wantErr)
				}
				continue loop
			}
		}
		if err := w.close(); err != tc.wantErr {
			t.Errorf("%s: got close error %v, want %v",
				tc.desc, err, tc.wantErr)
			continue
		}
		if rec.Code != tc.wantCode {
			t.Errorf("%s: got HTTP status code %d, want %d\n",
				tc.desc, rec.Code, tc.wantCode)
			continue
		}
		gotXML := rec.Body.String()
		eq, err := n.equalXML(strings.NewReader(gotXML), strings.NewReader(tc.wantXML))
		if err != nil {
			t.Errorf("%s: equalXML: %v", tc.desc, err)
			continue
		}
		if !eq {
			t.Errorf("%s: XML body\ngot  %s\nwant %s", tc.desc, gotXML, tc.wantXML)
		}
	}
}

func TestReadProppatch(t *testing.T) {
	ppStr := func(pps []Proppatch) string {
		var outer []string
		for _, pp := range pps {
			var inner []string
			for _, p := range pp.Props {
				inner = append(inner, fmt.Sprintf("{XMLName: %q, Lang: %q, InnerXML: %q}",
					p.XMLName, p.Lang, p.InnerXML))
			}
			outer = append(outer, fmt.Sprintf("{Remove: %t, Props: [%s]}",
				pp.Remove, strings.Join(inner, ", ")))
		}
		return "[" + strings.Join(outer, ", ") + "]"
	}

	testCases := []struct {
		desc       string
		input      string
		wantPP     []Proppatch
		wantStatus int
	}{{
		desc: "proppatch: section 9.2 (with simple property value)",
		input: `` +
			`<?xml version="1.0" encoding="utf-8" ?>` +
			`<D:propertyupdate xmlns:D="DAV:"` +
			`                  xmlns:Z="http://ns.example.com/z/">` +
			`    <D:set>` +
			`         <D:prop><Z:Authors>somevalue</Z:Authors></D:prop>` +
			`    </D:set>` +
			`    <D:remove>` +
			`         <D:prop><Z:Copyright-Owner/></D:prop>` +
			`    </D:remove>` +
			`</D:propertyupdate>`,
		wantPP: []Proppatch{{
			Props: []Property{{
				xml.Name{Space: "http://ns.example.com/z/", Local: "Authors"},
				"",
				[]byte(`somevalue`),
			}},
		}, {
			Remove: true,
			Props: []Property{{
				xml.Name{Space: "http://ns.example.com/z/", Local: "Copyright-Owner"},
				"",
				nil,
			}},
		}},
	}, {
		desc: "proppatch: lang attribute on prop",
		input: `` +
			`<?xml version="1.0" encoding="utf-8" ?>` +
			`<D:propertyupdate xmlns:D="DAV:">` +
			`    <D:set>` +
			`         <D:prop xml:lang="en">` +
			`              <foo xmlns="http://example.com/ns"/>` +
			`         </D:prop>` +
			`    </D:set>` +
			`</D:propertyupdate>`,
		wantPP: []Proppatch{{
			Props: []Property{{
				xml.Name{Space: "http://example.com/ns", Local: "foo"},
				"en",
				nil,
			}},
		}},
	}, {
		desc: "bad: remove with value",
		input: `` +
			`<?xml version="1.0" encoding="utf-8" ?>` +
			`<D:propertyupdate xmlns:D="DAV:"` +
			`                  xmlns:Z="http://ns.example.com/z/">` +
			`    <D:remove>` +
			`         <D:prop>` +
			`              <Z:Authors>` +
			`              <Z:Author>Jim Whitehead</Z:Author>` +
			`              </Z:Authors>` +
			`         </D:prop>` +
			`    </D:remove>` +
			`</D:propertyupdate>`,
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "bad: empty propertyupdate",
		input: `` +
			`<?xml version="1.0" encoding="utf-8" ?>` +
			`<D:propertyupdate xmlns:D="DAV:"` +
			`</D:propertyupdate>`,
		wantStatus: http.StatusBadRequest,
	}, {
		desc: "bad: empty prop",
		input: `` +
			`<?xml version="1.0" encoding="utf-8" ?>` +
			`<D:propertyupdate xmlns:D="DAV:"` +
			`                  xmlns:Z="http://ns.example.com/z/">` +
			`    <D:remove>` +
			`        <D:prop/>` +
			`    </D:remove>` +
			`</D:propertyupdate>`,
		wantStatus: http.StatusBadRequest,
	}}

	for _, tc := range testCases {
		pp, status, err := readProppatch(strings.NewReader(tc.input))
		if tc.wantStatus != 0 {
			if err == nil {
				t.Errorf("%s: got nil error, want non-nil", tc.desc)
				continue
			}
		} else if err != nil {
			t.Errorf("%s: %v", tc.desc, err)
			continue
		}
		if status != tc.wantStatus {
			t.Errorf("%s: got status %d, want %d", tc.desc, status, tc.wantStatus)
			continue
		}
		if !reflect.DeepEqual(pp, tc.wantPP) || status != tc.wantStatus {
			t.Errorf("%s: proppatch\ngot  %v\nwant %v", tc.desc, ppStr(pp), ppStr(tc.wantPP))
		}
	}
}

func TestUnmarshalXMLValue(t *testing.T) {
	testCases := []struct {
		desc    string
		input   string
		wantVal string
	}{{
		desc:    "simple char data",
		input:   "<root>foo</root>",
		wantVal: "foo",
	}, {
		desc:    "empty element",
		input:   "<root><foo/></root>",
		wantVal: "<foo/>",
	}, {
		desc:    "preserve namespace",
		input:   `<root><foo xmlns="bar"/></root>`,
		wantVal: `<foo xmlns="bar"/>`,
	}, {
		desc:    "preserve root element namespace",
		input:   `<root xmlns:bar="bar"><bar:foo/></root>`,
		wantVal: `<foo xmlns="bar"/>`,
	}, {
		desc:    "preserve whitespace",
		input:   "<root>  \t </root>",
		wantVal: "  \t ",
	}, {
		desc:    "preserve mixed content",
		input:   `<root xmlns="bar">  <foo>a<bam xmlns="baz"/> </foo> </root>`,
		wantVal: `  <foo xmlns="bar">a<bam xmlns="baz"/> </foo> `,
	}, {
		desc: "section 9.2",
		input: `` +
			`<Z:Authors xmlns:Z="http://ns.example.com/z/">` +
			`  <Z:Author>Jim Whitehead</Z:Author>` +
			`  <Z:Author>Roy Fielding</Z:Author>` +
			`</Z:Authors>`,
		wantVal: `` +
			`  <Author xmlns="http://ns.example.com/z/">Jim Whitehead</Author>` +
			`  <Author xmlns="http://ns.example.com/z/">Roy Fielding</Author>`,
	}, {
		desc: "section 4.3.1 (mixed content)",
		input: `` +
			`<x:author ` +
			`    xmlns:x='http://example.com/ns' ` +
			`    xmlns:D="DAV:">` +
			`  <x:name>Jane Doe</x:name>` +
			`  <!-- Jane's contact info -->` +
			`  <x:uri type='email'` +
			`         added='2005-11-26'>mailto:jane.doe@example.com</x:uri>` +
			`  <x:uri type='web'` +
			`         added='2005-11-27'>http://www.example.com</x:uri>` +
			`  <x:notes xmlns:h='http://www.w3.org/1999/xhtml'>` +
			`    Jane has been working way <h:em>too</h:em> long on the` +
			`    long-awaited revision of <![CDATA[<RFC2518>]]>.` +
			`  </x:notes>` +
			`</x:author>`,
		wantVal: `` +
			`  <name xmlns="http://example.com/ns">Jane Doe</name>` +
			`  ` +
			`  <uri type='email'` +
			`       xmlns="http://example.com/ns" ` +
			`       added='2005-11-26'>mailto:jane.doe@example.com</uri>` +
			`  <uri added='2005-11-27'` +
			`       type='web'` +
			`       xmlns="http://example.com/ns">http://www.example.com</uri>` +
			`  <notes xmlns="http://example.com/ns" ` +
			`         xmlns:h="http://www.w3.org/1999/xhtml">` +
			`    Jane has been working way <h:em>too</h:em> long on the` +
			`    long-awaited revision of &lt;RFC2518&gt;.` +
			`  </notes>`,
	}}

	var n xmlNormalizer
	for _, tc := range testCases {
		d := ixml.NewDecoder(strings.NewReader(tc.input))
		var v xmlValue
		if err := d.Decode(&v); err != nil {
			t.Errorf("%s: got error %v, want nil", tc.desc, err)
			continue
		}
		eq, err := n.equalXML(bytes.NewReader(v), strings.NewReader(tc.wantVal))
		if err != nil {
			t.Errorf("%s: equalXML: %v", tc.desc, err)
			continue
		}
		if !eq {
			t.Errorf("%s:\ngot  %s\nwant %s", tc.desc, string(v), tc.wantVal)
		}
	}
}

// xmlNormalizer normalizes XML.
type xmlNormalizer struct {
	// omitWhitespace instructs to ignore whitespace between element tags.
	omitWhitespace bool
	// omitComments instructs to ignore XML comments.
	omitComments bool
}

// normalize writes the normalized XML content of r to w. It applies the
// following rules
//
//     * Rename namespace prefixes according to an internal heuristic.
//     * Remove unnecessary namespace declarations.
//     * Sort attributes in XML start elements in lexical order of their
//       fully qualified name.
//     * Remove XML directives and processing instructions.
//     * Remove CDATA between XML tags that only contains whitespace, if
//       instructed to do so.
//     * Remove comments, if instructed to do so.
//
func (n *xmlNormalizer) normalize(w io.Writer, r io.Reader) error {
	d := ixml.NewDecoder(r)
	e := ixml.NewEncoder(w)
	for {
		t, err := d.Token()
		if err != nil {
			if t == nil && err == io.EOF {
				break
			}
			return err
		}
		switch val := t.(type) {
		case ixml.Directive, ixml.ProcInst:
			continue
		case ixml.Comment:
			if n.omitComments {
				continue
			}
		case ixml.CharData:
			if n.omitWhitespace && len(bytes.TrimSpace(val)) == 0 {
				continue
			}
		case ixml.StartElement:
			start, _ := ixml.CopyToken(val).(ixml.StartElement)
			attr := start.Attr[:0]
			for _, a := range start.Attr {
				if a.Name.Space == "xmlns" || a.Name.Local == "xmlns" {
					continue
				}
				attr = append(attr, a)
			}
			sort.Sort(byName(attr))
			start.Attr = attr
			t = start
		}
		err = e.EncodeToken(t)
		if err != nil {
			return err
		}
	}
	return e.Flush()
}

// equalXML tests for equality of the normalized XML contents of a and b.
func (n *xmlNormalizer) equalXML(a, b io.Reader) (bool, error) {
	var buf bytes.Buffer
	if err := n.normalize(&buf, a); err != nil {
		return false, err
	}
	normA := buf.String()
	buf.Reset()
	if err := n.normalize(&buf, b); err != nil {
		return false, err
	}
	normB := buf.String()
	return normA == normB, nil
}

type byName []ixml.Attr

func (a byName) Len() int      { return len(a) }
func (a byName) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a byName) Less(i, j int) bool {
	if a[i].Name.Space != a[j].Name.Space {
		return a[i].Name.Space < a[j].Name.Space
	}
	return a[i].Name.Local < a[j].Name.Local
}
