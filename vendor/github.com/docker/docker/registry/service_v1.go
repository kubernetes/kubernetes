package registry

import "net/url"

func (s *DefaultService) lookupV1Endpoints(hostname string) (endpoints []APIEndpoint, err error) {
	if hostname == DefaultNamespace || hostname == DefaultV2Registry.Host || hostname == IndexHostname {
		return []APIEndpoint{}, nil
	}

	tlsConfig, err := s.tlsConfig(hostname)
	if err != nil {
		return nil, err
	}

	endpoints = []APIEndpoint{
		{
			URL: &url.URL{
				Scheme: "https",
				Host:   hostname,
			},
			Version:      APIVersion1,
			TrimHostname: true,
			TLSConfig:    tlsConfig,
		},
	}

	if tlsConfig.InsecureSkipVerify {
		endpoints = append(endpoints, APIEndpoint{ // or this
			URL: &url.URL{
				Scheme: "http",
				Host:   hostname,
			},
			Version:      APIVersion1,
			TrimHostname: true,
			// used to check if supposed to be secure via InsecureSkipVerify
			TLSConfig: tlsConfig,
		})
	}
	return endpoints, nil
}
