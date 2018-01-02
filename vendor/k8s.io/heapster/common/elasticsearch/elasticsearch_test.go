// Copyright 2015 Google Inc. All Rights Reserved.
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

package elasticsearch

import (
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/olivere/elastic"
)

func TestCreateElasticSearchConfig(t *testing.T) {
	url, err := url.Parse("?nodes=https://foo.com:20468&nodes=https://bar.com:20468&esUserName=test&esUserSecret=password&maxRetries=10&startupHealthcheckTimeout=30&sniff=false&healthCheck=false")
	if err != nil {
		t.Fatalf("Error when parsing URL: %s", err.Error())
	}

	config, err := CreateElasticSearchConfig(url)
	if err != nil {
		t.Fatalf("Error when creating config: %s", err.Error())
	}

	expectedClient, err := elastic.NewClient(
		elastic.SetURL("https://foo.com:20468", "https://bar.com:20468"),
		elastic.SetBasicAuth("test", "password"),
		elastic.SetMaxRetries(10),
		elastic.SetHealthcheckTimeoutStartup(30*time.Second),
		elastic.SetSniff(false), elastic.SetHealthcheck(false))

	if err != nil {
		t.Fatalf("Error when creating client: %s", err.Error())
	}

	actualClientRefl := reflect.ValueOf(config.EsClient).Elem()
	expectedClientRefl := reflect.ValueOf(expectedClient).Elem()

	if actualClientRefl.FieldByName("basicAuthUsername").String() != expectedClientRefl.FieldByName("basicAuthUsername").String() {
		t.Fatalf("basicAuthUsername is not equal")
	}
	if actualClientRefl.FieldByName("basicAuthUsername").String() != expectedClientRefl.FieldByName("basicAuthUsername").String() {
		t.Fatalf("basicAuthUsername is not equal")
	}
	if actualClientRefl.FieldByName("maxRetries").Int() != expectedClientRefl.FieldByName("maxRetries").Int() {
		t.Fatalf("maxRetries is not equal")
	}
	if actualClientRefl.FieldByName("healthcheckTimeoutStartup").Int() != expectedClientRefl.FieldByName("healthcheckTimeoutStartup").Int() {
		t.Fatalf("healthcheckTimeoutStartup is not equal")
	}
	if actualClientRefl.FieldByName("snifferEnabled").Bool() != expectedClientRefl.FieldByName("snifferEnabled").Bool() {
		t.Fatalf("snifferEnabled is not equal")
	}
	if actualClientRefl.FieldByName("healthcheckEnabled").Bool() != expectedClientRefl.FieldByName("healthcheckEnabled").Bool() {
		t.Fatalf("healthcheckEnabled is not equal")
	}
}
