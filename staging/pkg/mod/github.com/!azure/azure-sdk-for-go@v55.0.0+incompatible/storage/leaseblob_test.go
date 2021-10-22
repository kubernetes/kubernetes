package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import chk "gopkg.in/check.v1"

type LeaseBlobSuite struct{}

var _ = chk.Suite(&LeaseBlobSuite{})

func (s *LeaseBlobSuite) TestAcquireLeaseWithNoProposedLeaseID(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	_, err := b.AcquireLease(30, "", nil)
	c.Assert(err, chk.IsNil)
}

func (s *LeaseBlobSuite) TestAcquireLeaseWithProposedLeaseID(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	proposedLeaseID := "dfe6dde8-68d5-4910-9248-c97c61768fea"
	leaseID, err := b.AcquireLease(30, proposedLeaseID, nil)
	c.Assert(err, chk.IsNil)
	c.Assert(leaseID, chk.Equals, proposedLeaseID)
}

func (s *LeaseBlobSuite) TestAcquireLeaseWithBadProposedLeaseID(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	proposedLeaseID := "badbadbad"
	_, err := b.AcquireLease(30, proposedLeaseID, nil)
	c.Assert(err, chk.NotNil)
}

func (s *LeaseBlobSuite) TestAcquireInfiniteLease(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	proposedLeaseID := "dfe6dde8-68d5-4910-9248-c97c61768fea"
	_, err := b.AcquireLease(-1, proposedLeaseID, nil)
	c.Assert(err, chk.IsNil)
}

func (s *LeaseBlobSuite) TestRenewLeaseSuccessful(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	proposedLeaseID := "dfe6dde8-68d5-4910-9248-c97c61768fea"
	leaseID, err := b.AcquireLease(30, proposedLeaseID, nil)
	c.Assert(err, chk.IsNil)

	err = b.RenewLease(leaseID, nil)
	c.Assert(err, chk.IsNil)
}

func (s *LeaseBlobSuite) TestRenewLeaseAgainstNoCurrentLease(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	badLeaseID := "Golang rocks on Azure"
	err := b.RenewLease(badLeaseID, nil)
	c.Assert(err, chk.NotNil)
}

func (s *LeaseBlobSuite) TestChangeLeaseSuccessful(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)
	proposedLeaseID := "dfe6dde8-68d5-4910-9248-c97c61768fea"
	leaseID, err := b.AcquireLease(30, proposedLeaseID, nil)
	c.Assert(err, chk.IsNil)

	newProposedLeaseID := "dfe6dde8-68d5-4910-9248-c97c61768fbb"
	newLeaseID, err := b.ChangeLease(leaseID, newProposedLeaseID, nil)
	c.Assert(err, chk.IsNil)
	c.Assert(newLeaseID, chk.Equals, newProposedLeaseID)
}

func (s *LeaseBlobSuite) TestChangeLeaseNotSuccessfulbadProposedLeaseID(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)
	proposedLeaseID := "dfe6dde8-68d5-4910-9248-c97c61768fea"
	leaseID, err := b.AcquireLease(30, proposedLeaseID, nil)
	c.Assert(err, chk.IsNil)

	newProposedLeaseID := "1f812371-a41d-49e6-b123-f4b542e"
	_, err = b.ChangeLease(leaseID, newProposedLeaseID, nil)
	c.Assert(err, chk.NotNil)
}

func (s *LeaseBlobSuite) TestReleaseLeaseSuccessful(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)
	proposedLeaseID := "dfe6dde8-68d5-4910-9248-c97c61768fea"
	leaseID, err := b.AcquireLease(30, proposedLeaseID, nil)
	c.Assert(err, chk.IsNil)

	err = b.ReleaseLease(leaseID, nil)
	c.Assert(err, chk.IsNil)
}

func (s *LeaseBlobSuite) TestReleaseLeaseNotSuccessfulBadLeaseID(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)
	proposedLeaseID := "dfe6dde8-68d5-4910-9248-c97c61768fea"
	_, err := b.AcquireLease(30, proposedLeaseID, nil)
	c.Assert(err, chk.IsNil)

	err = b.ReleaseLease("badleaseid", nil)
	c.Assert(err, chk.NotNil)
}

func (s *LeaseBlobSuite) TestBreakLeaseSuccessful(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	proposedLeaseID := "dfe6dde8-68d5-4910-9248-c97c61768fea"
	_, err := b.AcquireLease(30, proposedLeaseID, nil)
	c.Assert(err, chk.IsNil)

	_, err = b.BreakLease(nil)
	c.Assert(err, chk.IsNil)
}
