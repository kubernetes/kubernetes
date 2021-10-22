// Copyright 2018 Microsoft Corporation
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

package testdata

import (
	"context"
	"net/http"

	"github.com/Azure/azure-sdk-for-go/version"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
)

// content lifted from redis

const (
	DefaultBaseURI = "https://management.azure.com"
)

type DayOfWeek string

const (
	Everyday  DayOfWeek = "Everyday"
	Friday    DayOfWeek = "Friday"
	Monday    DayOfWeek = "Monday"
	Saturday  DayOfWeek = "Saturday"
	Sunday    DayOfWeek = "Sunday"
	Thursday  DayOfWeek = "Thursday"
	Tuesday   DayOfWeek = "Tuesday"
	Wednesday DayOfWeek = "Wednesday"
	Weekend   DayOfWeek = "Weekend"
)

type KeyType string

const (
	Primary   KeyType = "Primary"
	Secondary KeyType = "Secondary"
)

type BaseClient struct {
	autorest.Client
	BaseURI        string
	SubscriptionID string
}

type Client struct {
	BaseClient
}

func DoNothing() {
}

func DoNothingWithParam(foo int) {
}

func New(subscriptionID string) BaseClient {
	return NewWithBaseURI(DefaultBaseURI, subscriptionID)
}

func NewWithBaseURI(baseURI string, subscriptionID string) BaseClient {
	return BaseClient{
		Client:         autorest.NewClientWithUserAgent(UserAgent()),
		BaseURI:        baseURI,
		SubscriptionID: subscriptionID,
	}
}

func UserAgent() string {
	return "Azure-SDK-For-Go/" + version.Number + " redis/2017-10-01"
}

type CreateParameters struct {
	*CreateProperties `json:"properties,omitempty"`
	Zones             *[]string          `json:"zones,omitempty"`
	Location          *string            `json:"location,omitempty"`
	Tags              map[string]*string `json:"tags"`
}

func (cp CreateParameters) MarshalJSON() ([]byte, error) {
	return nil, nil
}

func (cp *CreateParameters) UnmarshalJSON(body []byte) error {
	return nil
}

type CreateProperties struct {
	SubnetID           *string            `json:"subnetId,omitempty"`
	StaticIP           *string            `json:"staticIP,omitempty"`
	RedisConfiguration map[string]*string `json:"redisConfiguration"`
	EnableNonSslPort   *bool              `json:"enableNonSslPort,omitempty"`
	TenantSettings     map[string]*string `json:"tenantSettings"`
	ShardCount         *int32             `json:"shardCount,omitempty"`
}

type DeleteFuture struct {
	azure.Future
	req *http.Request
}

func (future DeleteFuture) Result(client Client) (ar autorest.Response, err error) {
	return
}

type ListResult struct {
	autorest.Response `json:"-"`
	Value             *[]ResourceType `json:"value,omitempty"`
	NextLink          *string         `json:"nextLink,omitempty"`
}

func (lr ListResult) IsEmpty() bool {
	return lr.Value == nil || len(*lr.Value) == 0
}

type ListResultPage struct {
	fn func(ListResult) (ListResult, error)
	lr ListResult
}

func (page *ListResultPage) Next() error {
	return nil
}

func (page ListResultPage) NotDone() bool {
	return !page.lr.IsEmpty()
}

func (page ListResultPage) Response() ListResult {
	return page.lr
}

func (page ListResultPage) Values() []ResourceType {
	return *page.lr.Value
}

type ResourceType struct {
	autorest.Response `json:"-"`
	Zones             *[]string          `json:"zones,omitempty"`
	Tags              map[string]*string `json:"tags"`
	Location          *string            `json:"location,omitempty"`
	ID                *string            `json:"id,omitempty"`
	Name              *string            `json:"name,omitempty"`
	Type              *string            `json:"type,omitempty"`
}

func (client Client) Delete(ctx context.Context, resourceGroupName string, name string) (result DeleteFuture, err error) {
	return
}

func (client Client) DeletePreparer(ctx context.Context, resourceGroupName string, name string) (*http.Request, error) {
	const APIVersion = "2017-10-01"
	return nil, nil
}

func (client Client) DeleteSender(req *http.Request) (future DeleteFuture, err error) {
	return
}

func (client Client) DeleteResponder(resp *http.Response) (result autorest.Response, err error) {
	return
}

func (client Client) List(ctx context.Context) (result ListResultPage, err error) {
	return
}

func (client Client) ListPreparer(ctx context.Context) (*http.Request, error) {
	const APIVersion = "2017-10-01"
	return nil, nil
}

func (client Client) ListSender(req *http.Request) (*http.Response, error) {
	return nil, nil
}

func (client Client) ListResponder(resp *http.Response) (result ListResult, err error) {
	return
}

func (client Client) listNextResults(lastResults ListResult) (result ListResult, err error) {
	return
}

type SomeInterface interface {
	One()
	Two(bool)
}

type AnotherInterface interface {
	One()
}
