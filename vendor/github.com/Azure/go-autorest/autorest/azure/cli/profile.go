package cli

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
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/dimchansky/utfbom"
	"github.com/mitchellh/go-homedir"
)

// Profile represents a Profile from the Azure CLI
type Profile struct {
	InstallationID string         `json:"installationId"`
	Subscriptions  []Subscription `json:"subscriptions"`
}

// Subscription represents a Subscription from the Azure CLI
type Subscription struct {
	EnvironmentName string `json:"environmentName"`
	ID              string `json:"id"`
	IsDefault       bool   `json:"isDefault"`
	Name            string `json:"name"`
	State           string `json:"state"`
	TenantID        string `json:"tenantId"`
}

// ProfilePath returns the path where the Azure Profile is stored from the Azure CLI
func ProfilePath() (string, error) {
	return homedir.Expand("~/.azure/azureProfile.json")
}

// LoadProfile restores a Profile object from a file located at 'path'.
func LoadProfile(path string) (result Profile, err error) {
	var contents []byte
	contents, err = ioutil.ReadFile(path)
	if err != nil {
		err = fmt.Errorf("failed to open file (%s) while loading token: %v", path, err)
		return
	}
	reader := utfbom.SkipOnly(bytes.NewReader(contents))

	dec := json.NewDecoder(reader)
	if err = dec.Decode(&result); err != nil {
		err = fmt.Errorf("failed to decode contents of file (%s) into a Profile representation: %v", path, err)
		return
	}

	return
}
