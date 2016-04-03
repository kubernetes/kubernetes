// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bigquery

import (
	"fmt"

	"golang.org/x/net/context"
	bq "google.golang.org/api/bigquery/v2"
)

type copyOption interface {
	customizeCopy(conf *bq.JobConfigurationTableCopy, projectID string)
}

func (c *Client) cp(ctx context.Context, dst *Table, src Tables, options []Option) (*Job, error) {
	job, options := initJobProto(c.projectID, options)
	payload := &bq.JobConfigurationTableCopy{}

	dst.customizeCopyDst(payload, c.projectID)
	src.customizeCopySrc(payload, c.projectID)

	for _, opt := range options {
		o, ok := opt.(copyOption)
		if !ok {
			return nil, fmt.Errorf("option (%#v) not applicable to dst/src pair: dst: %T ; src: %T", opt, dst, src)
		}
		o.customizeCopy(payload, c.projectID)
	}

	job.Configuration = &bq.JobConfiguration{
		Copy: payload,
	}
	return c.service.insertJob(ctx, job, c.projectID)
}
