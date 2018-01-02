// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package photon

import (
	"bytes"
	"crypto/tls"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/vmware/photon-controller-go-sdk/photon/internal/mocks"
)

type MockTasksPage struct {
	Items            []Task `json:"items"`
	NextPageLink     string `json:"nextPageLink"`
	PreviousPageLink string `json:"previousPageLink"`
}

type MockAvailZonesPage struct {
	Items            []AvailabilityZone `json:"items"`
	NextPageLink     string             `json:"nextPageLink"`
	PreviousPageLink string             `json:"previousPageLink"`
}

type MockProjectsPage struct {
	Items            []ProjectCompact `json:"items"`
	NextPageLink     string           `json:"nextPageLink"`
	PreviousPageLink string           `json:"previousPageLink"`
}

type MockResourceTicketsPage struct {
	Items            []ResourceTicket `json:"items"`
	NextPageLink     string           `json:"nextPageLink"`
	PreviousPageLink string           `json:"previousPageLink"`
}

type MockTenantsPage struct {
	Items            []Tenant `json:"items"`
	NextPageLink     string   `json:"nextPageLink"`
	PreviousPageLink string   `json:"previousPageLink"`
}

type MockTenantPage struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type MockVmsPage struct {
	Items            []VM   `json:"items"`
	NextPageLink     string `json:"nextPageLink"`
	PreviousPageLink string `json:"previousPageLink"`
}

type MockFlavorsPage struct {
	Items            []Flavor `json:"items"`
	NextPageLink     string   `json:"nextPageLink"`
	PreviousPageLink string   `json:"previousPageLink"`
}

type MockNetworksPage struct {
	Items            []Network `json:"items"`
	NextPageLink     string    `json:"nextPageLink"`
	PreviousPageLink string    `json:"previousPageLink"`
}

type MockVirtualSubnetsPage struct {
	Items            []VirtualSubnet `json:"items"`
	NextPageLink     string          `json:"nextPageLink"`
	PreviousPageLink string          `json:"previousPageLink"`
}

type MockServicesPage struct {
	Items            []Service `json:"items"`
	NextPageLink     string    `json:"nextPageLink"`
	PreviousPageLink string    `json:"previousPageLink"`
}

type MockImagesPage struct {
	Items            []Image `json:"items"`
	NextPageLink     string  `json:"nextPageLink"`
	PreviousPageLink string  `json:"previousPageLink"`
}

type MockHostsPage struct {
	Items            []Host `json:"items"`
	NextPageLink     string `json:"nextPageLink"`
	PreviousPageLink string `json:"previousPageLink"`
}

func testSetup() (server *mocks.Server, client *Client) {
	// If TEST_ENDPOINT env var is set, return an empty server and point
	// the client to TEST_ENDPOINT. This lets us run tests as integration tests
	var uri string
	if os.Getenv("TEST_ENDPOINT") != "" {
		server = &mocks.Server{}
		uri = os.Getenv("TEST_ENDPOINT")
	} else {
		server = mocks.NewTestServer()
		uri = server.HttpServer.URL
	}

	options := &ClientOptions{
		IgnoreCertificate: true,
	}
	if os.Getenv("API_ACCESS_TOKEN") != "" {
		options.TokenOptions.AccessToken = os.Getenv("API_ACCESS_TOKEN")
	}

	transport := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: options.IgnoreCertificate,
		},
	}

	httpClient := &http.Client{Transport: transport}
	client = NewTestClient(uri, options, httpClient)
	return
}

func createMockStep(operation, state string) Step {
	return Step{State: state, Operation: operation}
}

func createMockTask(operation, state string, steps ...Step) *Task {
	return &Task{Operation: operation, State: state, Steps: steps}
}

func createMockTasksPage(tasks ...Task) *MockTasksPage {
	tasksPage := MockTasksPage{
		Items:            tasks,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &tasksPage
}

func createMockAvailZonesPage(availZones ...AvailabilityZone) *MockAvailZonesPage {
	availZonesPage := MockAvailZonesPage{
		Items:            availZones,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &availZonesPage
}

func createMockProjectsPage(projects ...ProjectCompact) *MockProjectsPage {
	projectsPage := MockProjectsPage{
		Items:            projects,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &projectsPage
}

func createMockResourceTicketsPage(resourceTickets ...ResourceTicket) *MockResourceTicketsPage {
	resourceTicketsPage := MockResourceTicketsPage{
		Items:            resourceTickets,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &resourceTicketsPage
}

func createMockTenantsPage(tenants ...Tenant) *MockTenantsPage {
	tenantsPage := MockTenantsPage{
		Items:            tenants,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &tenantsPage
}

func createMockTenantPage() *MockTenantPage {
	tenantPage := MockTenantPage{
		ID:   "12345",
		Name: "TestTenant",
	}
	return &tenantPage
}

func createMockVmsPage(vms ...VM) *MockVmsPage {
	vmsPage := MockVmsPage{
		Items:            vms,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &vmsPage
}

func createMockFlavorsPage(flavors ...Flavor) *MockFlavorsPage {
	flavorsPage := MockFlavorsPage{
		Items:            flavors,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &flavorsPage
}

func createMockNetworksPage(networks ...Network) *MockNetworksPage {
	networksPage := MockNetworksPage{
		Items:            networks,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &networksPage
}

func createMockVirtualSubnetsPage(networks ...VirtualSubnet) *MockVirtualSubnetsPage {
	virtualSubnetsPage := MockVirtualSubnetsPage{
		Items:            networks,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &virtualSubnetsPage
}

func createMockServicesPage(services ...Service) *MockServicesPage {
	servicesPage := MockServicesPage{
		Items:            services,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &servicesPage
}

func createMockImagesPage(images ...Image) *MockImagesPage {
	imagesPage := MockImagesPage{
		Items:            images,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &imagesPage
}

func createMockHostsPage(hosts ...Host) *MockHostsPage {
	hostsPage := MockHostsPage{
		Items:            hosts,
		NextPageLink:     "",
		PreviousPageLink: "",
	}

	return &hostsPage
}

func createMockApiError(code string, message string, httpStatusCode int) *ApiError {
	apiError := ApiError{
		Code:           code,
		Message:        message,
		HttpStatusCode: httpStatusCode,
	}

	return &apiError
}

func createMockAuthInfo(server *mocks.Server) (mock *AuthInfo) {
	mock = &AuthInfo{
		Endpoint: "",
		Port:     0,
	}

	if server == nil {
		return
	}

	address, port, err := server.GetAddressAndPort()
	if err != nil {
		return
	}

	mock.Endpoint = address
	mock.Port = port

	return
}

var letters = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

func randomString(n int, prefixes ...string) string {
	rand.Seed(time.Now().UTC().UnixNano())
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}

	var buffer bytes.Buffer

	for i := 0; i < len(prefixes); i++ {
		buffer.WriteString(prefixes[i])
	}

	buffer.WriteString(string(b))
	return buffer.String()
}

func randomAddress() string {
	rand.Seed(time.Now().UTC().UnixNano())
	addr := strconv.Itoa(rand.Intn(256))
	for i := 0; i < 3; i++ {
		addr += "." + strconv.Itoa(rand.Intn(256))
	}
	return addr
}

func isRealAgent() bool {
	return os.Getenv("REAL_AGENT") != ""
}

func isIntegrationTest() bool {
	return os.Getenv("TEST_ENDPOINT") != ""
}
