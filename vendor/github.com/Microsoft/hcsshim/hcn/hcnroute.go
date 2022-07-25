package hcn

import (
	"encoding/json"
	"errors"

	"github.com/Microsoft/go-winio/pkg/guid"
	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/sirupsen/logrus"
)

// HostComputeRoute represents SDN routes.
type HostComputeRoute struct {
	ID                   string                  `json:"ID,omitempty"`
	HostComputeEndpoints []string                `json:",omitempty"`
	Setting              []SDNRoutePolicySetting `json:",omitempty"`
	SchemaVersion        SchemaVersion           `json:",omitempty"`
}

// ListRoutes makes a call to list all available routes.
func ListRoutes() ([]HostComputeRoute, error) {
	hcnQuery := defaultQuery()
	routes, err := ListRoutesQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	return routes, nil
}

// ListRoutesQuery makes a call to query the list of available routes.
func ListRoutesQuery(query HostComputeQuery) ([]HostComputeRoute, error) {
	queryJSON, err := json.Marshal(query)
	if err != nil {
		return nil, err
	}

	routes, err := enumerateRoutes(string(queryJSON))
	if err != nil {
		return nil, err
	}
	return routes, nil
}

// GetRouteByID returns the route specified by Id.
func GetRouteByID(routeID string) (*HostComputeRoute, error) {
	hcnQuery := defaultQuery()
	mapA := map[string]string{"ID": routeID}
	filter, err := json.Marshal(mapA)
	if err != nil {
		return nil, err
	}
	hcnQuery.Filter = string(filter)

	routes, err := ListRoutesQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	if len(routes) == 0 {
		return nil, RouteNotFoundError{RouteId: routeID}
	}
	return &routes[0], err
}

// Create Route.
func (route *HostComputeRoute) Create() (*HostComputeRoute, error) {
	logrus.Debugf("hcn::HostComputeRoute::Create id=%s", route.ID)

	jsonString, err := json.Marshal(route)
	if err != nil {
		return nil, err
	}

	logrus.Debugf("hcn::HostComputeRoute::Create JSON: %s", jsonString)
	route, hcnErr := createRoute(string(jsonString))
	if hcnErr != nil {
		return nil, hcnErr
	}
	return route, nil
}

// Delete Route.
func (route *HostComputeRoute) Delete() error {
	logrus.Debugf("hcn::HostComputeRoute::Delete id=%s", route.ID)

	existingRoute, _ := GetRouteByID(route.ID)

	if existingRoute != nil {
		if err := deleteRoute(route.ID); err != nil {
			return err
		}
	}

	return nil
}

// AddEndpoint add an endpoint to a route
// Since HCNRoute doesn't implement modify functionality, add operation is essentially delete and add
func (route *HostComputeRoute) AddEndpoint(endpoint *HostComputeEndpoint) (*HostComputeRoute, error) {
	logrus.Debugf("hcn::HostComputeRoute::AddEndpoint route=%s endpoint=%s", route.ID, endpoint.Id)

	err := route.Delete()
	if err != nil {
		return nil, err
	}

	// Add Endpoint to the Existing List
	route.HostComputeEndpoints = append(route.HostComputeEndpoints, endpoint.Id)

	return route.Create()
}

// RemoveEndpoint removes an endpoint from a route
// Since HCNRoute doesn't implement modify functionality, remove operation is essentially delete and add
func (route *HostComputeRoute) RemoveEndpoint(endpoint *HostComputeEndpoint) (*HostComputeRoute, error) {
	logrus.Debugf("hcn::HostComputeRoute::RemoveEndpoint route=%s endpoint=%s", route.ID, endpoint.Id)

	err := route.Delete()
	if err != nil {
		return nil, err
	}

	// Create a list of all the endpoints besides the one being removed
	i := 0
	for index, endpointReference := range route.HostComputeEndpoints {
		if endpointReference == endpoint.Id {
			i = index
			break
		}
	}

	route.HostComputeEndpoints = append(route.HostComputeEndpoints[0:i], route.HostComputeEndpoints[i+1:]...)
	return route.Create()
}

// AddRoute for the specified endpoints and SDN Route setting
func AddRoute(endpoints []HostComputeEndpoint, destinationPrefix string, nextHop string, needEncapsulation bool) (*HostComputeRoute, error) {
	logrus.Debugf("hcn::HostComputeRoute::AddRoute endpointId=%v, destinationPrefix=%v, nextHop=%v, needEncapsulation=%v", endpoints, destinationPrefix, nextHop, needEncapsulation)

	if len(endpoints) <= 0 {
		return nil, errors.New("Missing endpoints")
	}

	route := &HostComputeRoute{
		SchemaVersion: V2SchemaVersion(),
		Setting: []SDNRoutePolicySetting{
			{
				DestinationPrefix: destinationPrefix,
				NextHop:           nextHop,
				NeedEncap:         needEncapsulation,
			},
		},
	}

	for _, endpoint := range endpoints {
		route.HostComputeEndpoints = append(route.HostComputeEndpoints, endpoint.Id)
	}

	return route.Create()
}

func enumerateRoutes(query string) ([]HostComputeRoute, error) {
	// Enumerate all routes Guids
	var (
		resultBuffer *uint16
		routeBuffer  *uint16
	)
	hr := hcnEnumerateRoutes(query, &routeBuffer, &resultBuffer)
	if err := checkForErrors("hcnEnumerateRoutes", hr, resultBuffer); err != nil {
		return nil, err
	}

	routes := interop.ConvertAndFreeCoTaskMemString(routeBuffer)
	var routeIds []guid.GUID
	if err := json.Unmarshal([]byte(routes), &routeIds); err != nil {
		return nil, err
	}

	var outputRoutes []HostComputeRoute
	for _, routeGUID := range routeIds {
		route, err := getRoute(routeGUID, query)
		if err != nil {
			return nil, err
		}
		outputRoutes = append(outputRoutes, *route)
	}
	return outputRoutes, nil
}

func getRoute(routeGUID guid.GUID, query string) (*HostComputeRoute, error) {
	// Open routes.
	var (
		routeHandle      hcnRoute
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	hr := hcnOpenRoute(&routeGUID, &routeHandle, &resultBuffer)
	if err := checkForErrors("hcnOpenRoute", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query routes.
	hr = hcnQueryRouteProperties(routeHandle, query, &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryRouteProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close routes.
	hr = hcnCloseRoute(routeHandle)
	if err := checkForErrors("hcnCloseRoute", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeRoute
	var outputRoute HostComputeRoute
	if err := json.Unmarshal([]byte(properties), &outputRoute); err != nil {
		return nil, err
	}
	return &outputRoute, nil
}

func createRoute(settings string) (*HostComputeRoute, error) {
	// Create new route.
	var (
		routeHandle      hcnRoute
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	routeGUID := guid.GUID{}
	hr := hcnCreateRoute(&routeGUID, settings, &routeHandle, &resultBuffer)
	if err := checkForErrors("hcnCreateRoute", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query route.
	hcnQuery := defaultQuery()
	query, err := json.Marshal(hcnQuery)
	if err != nil {
		return nil, err
	}
	hr = hcnQueryRouteProperties(routeHandle, string(query), &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryRouteProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close Route.
	hr = hcnCloseRoute(routeHandle)
	if err := checkForErrors("hcnCloseRoute", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeRoute
	var outputRoute HostComputeRoute
	if err := json.Unmarshal([]byte(properties), &outputRoute); err != nil {
		return nil, err
	}
	return &outputRoute, nil
}

func deleteRoute(routeID string) error {
	routeGUID, err := guid.FromString(routeID)
	if err != nil {
		return errInvalidRouteID
	}
	var resultBuffer *uint16
	hr := hcnDeleteRoute(&routeGUID, &resultBuffer)
	if err := checkForErrors("hcnDeleteRoute", hr, resultBuffer); err != nil {
		return err
	}
	return nil
}
