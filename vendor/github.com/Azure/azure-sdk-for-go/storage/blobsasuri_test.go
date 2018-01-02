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
	"io/ioutil"
	"net/http"
	"net/url"
	"time"

	chk "gopkg.in/check.v1"
)

type BlobSASURISuite struct{}

var _ = chk.Suite(&BlobSASURISuite{})

var oldAPIVer = "2013-08-15"
var newerAPIVer = "2015-04-05"

func (s *BlobSASURISuite) TestGetBlobSASURI(c *chk.C) {
	api, err := NewClient("foo", dummyMiniStorageKey, DefaultBaseURL, oldAPIVer, true)
	c.Assert(err, chk.IsNil)
	cli := api.GetBlobService()
	cnt := cli.GetContainerReference("container")
	b := cnt.GetBlobReference("name")

	expectedParts := url.URL{
		Scheme: "https",
		Host:   "foo.blob.core.windows.net",
		Path:   "container/name",
		RawQuery: url.Values{
			"sv":  {oldAPIVer},
			"sig": {"/OXG7rWh08jYwtU03GzJM0DHZtidRGpC6g69rSGm3I0="},
			"sr":  {"b"},
			"sp":  {"r"},
			"se":  {"0001-01-01T00:00:00Z"},
		}.Encode(),
	}

	sasuriOptions := BlobSASOptions{}
	sasuriOptions.Read = true
	sasuriOptions.UseHTTPS = true

	u, err := b.GetSASURI(sasuriOptions)
	c.Assert(err, chk.IsNil)
	sasParts, err := url.Parse(u)
	c.Assert(err, chk.IsNil)
	c.Assert(expectedParts.String(), chk.Equals, sasParts.String())
	c.Assert(expectedParts.Query(), chk.DeepEquals, sasParts.Query())
}

//Gets a SASURI for the entire container
func (s *BlobSASURISuite) TestGetBlobSASURIContainer(c *chk.C) {
	api, err := NewClient("foo", dummyMiniStorageKey, DefaultBaseURL, oldAPIVer, true)
	c.Assert(err, chk.IsNil)
	cli := api.GetBlobService()
	cnt := cli.GetContainerReference("container")

	expectedParts := url.URL{
		Scheme: "https",
		Host:   "foo.blob.core.windows.net",
		Path:   "container",
		RawQuery: url.Values{
			"sv":  {oldAPIVer},
			"sig": {"KMjYyQODKp6uK9EKR3yGhO2M84e1LfoztypU32kHj4s="},
			"sr":  {"c"},
			"sp":  {"r"},
			"se":  {"0001-01-01T00:00:00Z"},
		}.Encode(),
	}

	sasuriOptions := ContainerSASOptions{}
	sasuriOptions.Read = true

	u, err := cnt.GetSASURI(sasuriOptions)
	c.Assert(err, chk.IsNil)
	sasParts, err := url.Parse(u)
	c.Assert(err, chk.IsNil)
	c.Assert(expectedParts.String(), chk.Equals, sasParts.String())
	c.Assert(expectedParts.Query(), chk.DeepEquals, sasParts.Query())
}

func (s *BlobSASURISuite) TestGetBlobSASURIWithSignedIPAndProtocolValidAPIVersionPassed(c *chk.C) {
	api, err := NewClient("foo", dummyMiniStorageKey, DefaultBaseURL, newerAPIVer, true)
	c.Assert(err, chk.IsNil)
	cli := api.GetBlobService()
	cnt := cli.GetContainerReference("container")
	b := cnt.GetBlobReference("name")

	expectedParts := url.URL{
		Scheme: "https",
		Host:   "foo.blob.core.windows.net",
		Path:   "/container/name",
		RawQuery: url.Values{
			"sv":  {newerAPIVer},
			"sig": {"VBOYJmt89UuBRXrxNzmsCMoC+8PXX2yklV71QcL1BfM="},
			"sr":  {"b"},
			"sip": {"127.0.0.1"},
			"sp":  {"r"},
			"se":  {"0001-01-01T00:00:00Z"},
			"spr": {"https"},
		}.Encode(),
	}

	sasuriOptions := BlobSASOptions{}
	sasuriOptions.Read = true
	sasuriOptions.IP = "127.0.0.1"
	sasuriOptions.UseHTTPS = true

	u, err := b.GetSASURI(sasuriOptions)
	c.Assert(err, chk.IsNil)
	sasParts, err := url.Parse(u)
	c.Assert(err, chk.IsNil)
	c.Assert(sasParts.Query(), chk.DeepEquals, expectedParts.Query())
}

// Trying to use SignedIP and Protocol but using an older version of the API.
// Should ignore the signedIP/protocol and just use what the older version requires.
func (s *BlobSASURISuite) TestGetBlobSASURIWithSignedIPAndProtocolUsingOldAPIVersion(c *chk.C) {
	api, err := NewClient("foo", dummyMiniStorageKey, DefaultBaseURL, oldAPIVer, true)
	c.Assert(err, chk.IsNil)
	cli := api.GetBlobService()
	cnt := cli.GetContainerReference("container")
	b := cnt.GetBlobReference("name")

	expectedParts := url.URL{
		Scheme: "https",
		Host:   "foo.blob.core.windows.net",
		Path:   "/container/name",
		RawQuery: url.Values{
			"sv":  {oldAPIVer},
			"sig": {"/OXG7rWh08jYwtU03GzJM0DHZtidRGpC6g69rSGm3I0="},
			"sr":  {"b"},
			"sp":  {"r"},
			"se":  {"0001-01-01T00:00:00Z"},
		}.Encode(),
	}

	sasuriOptions := BlobSASOptions{}
	sasuriOptions.Read = true
	sasuriOptions.UseHTTPS = true

	u, err := b.GetSASURI(sasuriOptions)
	c.Assert(err, chk.IsNil)
	sasParts, err := url.Parse(u)
	c.Assert(err, chk.IsNil)
	c.Assert(expectedParts.String(), chk.Equals, sasParts.String())
	c.Assert(expectedParts.Query(), chk.DeepEquals, sasParts.Query())
}

func (s *BlobSASURISuite) TestBlobSASURICorrectness(c *chk.C) {
	cli := getBlobClient(c)
	simpleClient := &http.Client{}
	rec := cli.client.appendRecorder(c)
	simpleClient.Transport = rec
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	b := cnt.GetBlobReference(contentWithSpecialChars(5))
	defer cnt.Delete(nil)

	body := content(100)
	c.Assert(b.putSingleBlockBlob(body), chk.IsNil)

	sasuriOptions := BlobSASOptions{}
	sasuriOptions.Expiry = fixedTime.UTC().Add(time.Hour)
	sasuriOptions.Read = true

	sasURI, err := b.GetSASURI(sasuriOptions)
	c.Assert(err, chk.IsNil)

	resp, err := simpleClient.Get(sasURI)
	c.Assert(err, chk.IsNil)

	blobResp, err := ioutil.ReadAll(resp.Body)
	defer resp.Body.Close()
	c.Assert(err, chk.IsNil)

	c.Assert(resp.StatusCode, chk.Equals, http.StatusOK)
	c.Assert(string(blobResp), chk.Equals, string(body))

}

func (s *BlobSASURISuite) Test_blobSASStringToSign(c *chk.C) {
	_, err := blobSASStringToSign("2012-02-12", "CS", "SE", "SP", "", "", "", "", OverrideHeaders{})
	c.Assert(err, chk.NotNil) // not implemented SAS for versions earlier than 2013-08-15

	out, err := blobSASStringToSign("SP", "", "SE", "CS", "", "", "", oldAPIVer, OverrideHeaders{})
	c.Assert(err, chk.IsNil)
	c.Assert(out, chk.Equals, "SP\n\nSE\nCS\n\n2013-08-15\n\n\n\n\n")

	// check format for 2015-04-05 version
	out, err = blobSASStringToSign("SP", "", "SE", "CS", "", "127.0.0.1", "https,http", newerAPIVer, OverrideHeaders{})
	c.Assert(err, chk.IsNil)
	c.Assert(out, chk.Equals, "SP\n\nSE\n/blobCS\n\n127.0.0.1\nhttps,http\n2015-04-05\n\n\n\n\n")
}

func (s *BlobSASURISuite) TestGetBlobSASURIStorageEmulator(c *chk.C) {
	client, err := NewEmulatorClient()
	c.Assert(err, chk.IsNil)
	blobService := client.GetBlobService()
	container := blobService.GetContainerReference("testfolder")
	blob := container.GetBlobReference("testfile")
	options := BlobSASOptions{
		SASOptions: SASOptions{
			Expiry: time.Date(2017, 9, 30, 16, 0, 0, 0, time.UTC),
		},
		BlobServiceSASPermissions: BlobServiceSASPermissions{
			Write: true,
		},
	}
	url, err := blob.GetSASURI(options)
	c.Assert(err, chk.IsNil)
	c.Assert(url, chk.Equals, "http://127.0.0.1:10000/devstoreaccount1/testfolder/testfile?se=2017-09-30T16%3A00%3A00Z&sig=Tyrg2ccc0RXyRz5xfkcSVDvjjoRivygrGb%2ByTLf0jJY%3D&sp=w&sr=b&sv=2016-05-31")
}
