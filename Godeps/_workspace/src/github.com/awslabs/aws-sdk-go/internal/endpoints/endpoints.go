package endpoints

//go:generate go run ../model/cli/gen-endpoints/main.go endpoints.json endpoints_map.go

import "strings"

// EndpointForRegion returns an endpoint and its signing region for a service and region.
// if the service and region pair are not found endpoint and signingRegion will be empty.
func EndpointForRegion(svcName, region string) (endpoint, signingRegion string) {
	derivedKeys := []string{
		region + "/" + svcName,
		region + "/*",
		"*/" + svcName,
		"*/*",
	}

	for _, key := range derivedKeys {
		if val, ok := endpointsMap.Endpoints[key]; ok {
			ep := val.Endpoint
			ep = strings.Replace(ep, "{region}", region, -1)
			ep = strings.Replace(ep, "{service}", svcName, -1)

			endpoint = ep
			signingRegion = val.SigningRegion
			return
		}
	}
	return
}
