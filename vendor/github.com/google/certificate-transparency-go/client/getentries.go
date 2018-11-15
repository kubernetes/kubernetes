// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package client

import (
	"context"
	"errors"
	"strconv"

	ct "github.com/google/certificate-transparency-go"
	"github.com/google/certificate-transparency-go/x509"
)

// GetRawEntries exposes the /ct/v1/get-entries result with only the JSON parsing done.
func (c *LogClient) GetRawEntries(ctx context.Context, start, end int64) (*ct.GetEntriesResponse, error) {
	if end < 0 {
		return nil, errors.New("end should be >= 0")
	}
	if end < start {
		return nil, errors.New("start should be <= end")
	}

	params := map[string]string{
		"start": strconv.FormatInt(start, 10),
		"end":   strconv.FormatInt(end, 10),
	}
	if ctx == nil {
		ctx = context.TODO()
	}

	var resp ct.GetEntriesResponse
	httpRsp, body, err := c.GetAndParse(ctx, ct.GetEntriesPath, params, &resp)
	if err != nil {
		if httpRsp != nil {
			return nil, RspError{Err: err, StatusCode: httpRsp.StatusCode, Body: body}
		}
		return nil, err
	}

	return &resp, nil
}

// GetEntries attempts to retrieve the entries in the sequence [start, end] from the CT log server
// (RFC6962 s4.6) as parsed [pre-]certificates for convenience, held in a slice of ct.LogEntry structures.
// However, this does mean that any certificate parsing failures will cause a failure of the whole
// retrieval operation; for more robust retrieval of parsed certificates, use GetRawEntries() and invoke
// ct.LogEntryFromLeaf() on each individual entry.
func (c *LogClient) GetEntries(ctx context.Context, start, end int64) ([]ct.LogEntry, error) {
	resp, err := c.GetRawEntries(ctx, start, end)
	if err != nil {
		return nil, err
	}
	entries := make([]ct.LogEntry, len(resp.Entries))
	for i, entry := range resp.Entries {
		index := start + int64(i)
		logEntry, err := ct.LogEntryFromLeaf(index, &entry)
		if x509.IsFatal(err) {
			return nil, err
		}
		entries[i] = *logEntry
	}
	return entries, nil
}
