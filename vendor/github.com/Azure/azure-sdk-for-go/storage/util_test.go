package storage

import (
	"encoding/xml"
	"io/ioutil"
	"net/url"
	"strings"
	"time"

	chk "gopkg.in/check.v1"
)

func (s *StorageClientSuite) Test_timeRfc1123Formatted(c *chk.C) {
	now := time.Now().UTC()
	expectedLayout := "Mon, 02 Jan 2006 15:04:05 GMT"
	c.Assert(timeRfc1123Formatted(now), chk.Equals, now.Format(expectedLayout))
}

func (s *StorageClientSuite) Test_mergeParams(c *chk.C) {
	v1 := url.Values{
		"k1": {"v1"},
		"k2": {"v2"}}
	v2 := url.Values{
		"k1": {"v11"},
		"k3": {"v3"}}
	out := mergeParams(v1, v2)
	c.Assert(out.Get("k1"), chk.Equals, "v1")
	c.Assert(out.Get("k2"), chk.Equals, "v2")
	c.Assert(out.Get("k3"), chk.Equals, "v3")
	c.Assert(out["k1"], chk.DeepEquals, []string{"v1", "v11"})
}

func (s *StorageClientSuite) Test_prepareBlockListRequest(c *chk.C) {
	empty := []Block{}
	expected := `<?xml version="1.0" encoding="utf-8"?><BlockList></BlockList>`
	c.Assert(prepareBlockListRequest(empty), chk.DeepEquals, expected)

	blocks := []Block{{"foo", BlockStatusLatest}, {"bar", BlockStatusUncommitted}}
	expected = `<?xml version="1.0" encoding="utf-8"?><BlockList><Latest>foo</Latest><Uncommitted>bar</Uncommitted></BlockList>`
	c.Assert(prepareBlockListRequest(blocks), chk.DeepEquals, expected)
}

func (s *StorageClientSuite) Test_xmlUnmarshal(c *chk.C) {
	xml := `<?xml version="1.0" encoding="utf-8"?>
	<Blob>
		<Name>myblob</Name>
	</Blob>`
	var blob Blob
	body := ioutil.NopCloser(strings.NewReader(xml))
	c.Assert(xmlUnmarshal(body, &blob), chk.IsNil)
	c.Assert(blob.Name, chk.Equals, "myblob")
}

func (s *StorageClientSuite) Test_xmlMarshal(c *chk.C) {
	type t struct {
		XMLName xml.Name `xml:"S"`
		Name    string   `xml:"Name"`
	}

	b := t{Name: "myblob"}
	expected := `<S><Name>myblob</Name></S>`
	r, i, err := xmlMarshal(b)
	c.Assert(err, chk.IsNil)
	o, err := ioutil.ReadAll(r)
	c.Assert(err, chk.IsNil)
	out := string(o)
	c.Assert(out, chk.Equals, expected)
	c.Assert(i, chk.Equals, len(expected))
}
