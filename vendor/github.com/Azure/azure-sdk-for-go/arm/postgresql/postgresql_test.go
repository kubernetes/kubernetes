package postgresql

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
	"encoding/json"
	"testing"

	"github.com/Azure/go-autorest/autorest/to"
	chk "gopkg.in/check.v1"
)

// Hook up gocheck to testing
func Test(t *testing.T) { chk.TestingT(t) }

type Suite struct{}

var _ = chk.Suite(&Suite{})

var (
	body = `{
    "sku": {
        "name": "SkuName",
        "tier": "Basic",
        "capacity": 100
    },
    "properties": {
        "storageMB": 1024,
        "sslEnforcement": "Enabled",
        "createMode": "Default",
        "administratorLogin": "cloudsa",
        "administratorLoginPassword": "password"
    },
    "location": "OneBox",
    "tags": {
        "ElasticServer": "1"
    }
}`
	sfc = ServerForCreate{
		Location: to.StringPtr("OneBox"),
		Properties: ServerPropertiesForDefaultCreate{
			AdministratorLogin:         to.StringPtr("cloudsa"),
			AdministratorLoginPassword: to.StringPtr("password"),
			StorageMB:                  to.Int64Ptr(1024),
			SslEnforcement:             SslEnforcementEnumEnabled,
			CreateMode:                 CreateModeDefault,
		},
		Sku: &Sku{
			Name:     to.StringPtr("SkuName"),
			Tier:     Basic,
			Capacity: to.Int32Ptr(100),
		},
		Tags: &map[string]*string{
			"ElasticServer": to.StringPtr("1"),
		},
	}
)

func (s *Suite) TestUnmarshalServerForCreate(c *chk.C) {
	var obtained ServerForCreate
	err := json.Unmarshal([]byte(body), &obtained)
	c.Assert(err, chk.IsNil)
	c.Assert(obtained, chk.DeepEquals, sfc)
}

func (s *Suite) TestMarshalServerForCreate(c *chk.C) {
	b, err := json.MarshalIndent(sfc, "", "    ")
	c.Assert(err, chk.IsNil)
	c.Assert(string(b), chk.Equals, body)
}
