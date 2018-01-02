package storage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"net/url"
	"time"

	chk "gopkg.in/check.v1"
)

type QueueSASURISuite struct{}

var _ = chk.Suite(&QueueSASURISuite{})

var queueOldAPIVer = "2013-08-15"
var queueNewerAPIVer = "2015-04-05"

func (s *QueueSASURISuite) TestGetQueueSASURI(c *chk.C) {
	api, err := NewClient("foo", dummyMiniStorageKey, DefaultBaseURL, queueOldAPIVer, true)
	c.Assert(err, chk.IsNil)
	cli := api.GetQueueService()
	q := cli.GetQueueReference("name")

	expectedParts := url.URL{
		Scheme: "https",
		Host:   "foo.queue.core.windows.net",
		Path:   "name",
		RawQuery: url.Values{
			"sv":  {oldAPIVer},
			"sig": {"dYZ+elcEz3ZXEnTDKR5+RCrMzk0L7/ATWsemNzb36VM="},
			"sp":  {"p"},
			"se":  {"0001-01-01T00:00:00Z"},
		}.Encode()}

	options := QueueSASOptions{}
	options.Process = true
	options.Expiry = time.Time{}

	u, err := q.GetSASURI(options)
	c.Assert(err, chk.IsNil)
	sasParts, err := url.Parse(u)
	c.Assert(err, chk.IsNil)
	c.Assert(expectedParts.String(), chk.Equals, sasParts.String())
	c.Assert(expectedParts.Query(), chk.DeepEquals, sasParts.Query())
}

func (s *QueueSASURISuite) TestGetQueueSASURIWithSignedIPValidAPIVersionPassed(c *chk.C) {
	api, err := NewClient("foo", dummyMiniStorageKey, DefaultBaseURL, queueNewerAPIVer, true)
	c.Assert(err, chk.IsNil)
	cli := api.GetQueueService()
	q := cli.GetQueueReference("name")

	expectedParts := url.URL{
		Scheme: "https",
		Host:   "foo.queue.core.windows.net",
		Path:   "/name",
		RawQuery: url.Values{
			"sv":  {newerAPIVer},
			"sig": {"8uvfE93HdYxQ3xvt/CUN3S7sYEl1LcuHBC0oYoGDnfw="},
			"sip": {"127.0.0.1"},
			"sp":  {"p"},
			"se":  {"0001-01-01T00:00:00Z"},
			"spr": {"https,http"},
		}.Encode()}

	options := QueueSASOptions{}
	options.Process = true
	options.Expiry = time.Time{}
	options.IP = "127.0.0.1"

	u, err := q.GetSASURI(options)
	c.Assert(err, chk.IsNil)
	sasParts, err := url.Parse(u)
	c.Assert(err, chk.IsNil)
	c.Assert(sasParts.Query(), chk.DeepEquals, expectedParts.Query())
}

// Trying to use SignedIP but using an older version of the API.
// Should ignore the signedIP and just use what the older version requires.
func (s *QueueSASURISuite) TestGetQueueSASURIWithSignedIPUsingOldAPIVersion(c *chk.C) {
	api, err := NewClient("foo", dummyMiniStorageKey, DefaultBaseURL, oldAPIVer, true)
	c.Assert(err, chk.IsNil)
	cli := api.GetQueueService()
	q := cli.GetQueueReference("name")

	expectedParts := url.URL{
		Scheme: "https",
		Host:   "foo.queue.core.windows.net",
		Path:   "/name",
		RawQuery: url.Values{
			"sv":  {oldAPIVer},
			"sig": {"dYZ+elcEz3ZXEnTDKR5+RCrMzk0L7/ATWsemNzb36VM="},
			"sp":  {"p"},
			"se":  {"0001-01-01T00:00:00Z"},
		}.Encode()}

	options := QueueSASOptions{}
	options.Process = true
	options.Expiry = time.Time{}
	options.IP = "127.0.0.1"

	u, err := q.GetSASURI(options)
	c.Assert(err, chk.IsNil)
	sasParts, err := url.Parse(u)
	c.Assert(err, chk.IsNil)
	c.Assert(expectedParts.String(), chk.Equals, sasParts.String())
	c.Assert(expectedParts.Query(), chk.DeepEquals, sasParts.Query())
}
