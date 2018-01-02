package libcontainerd

import "strings"

// setupEnvironmentVariables converts a string array of environment variables
// into a map as required by the HCS. Source array is in format [v1=k1] [v2=k2] etc.
func setupEnvironmentVariables(a []string) map[string]string {
	r := make(map[string]string)
	for _, s := range a {
		arr := strings.SplitN(s, "=", 2)
		if len(arr) == 2 {
			r[arr[0]] = arr[1]
		}
	}
	return r
}

// Apply for a servicing option is a no-op.
func (s *ServicingOption) Apply(interface{}) error {
	return nil
}

// Apply for the flush option is a no-op.
func (f *FlushOption) Apply(interface{}) error {
	return nil
}

// Apply for the hypervisolation option is a no-op.
func (h *HyperVIsolationOption) Apply(interface{}) error {
	return nil
}

// Apply for the layer option is a no-op.
func (h *LayerOption) Apply(interface{}) error {
	return nil
}

// Apply for the network endpoints option is a no-op.
func (s *NetworkEndpointsOption) Apply(interface{}) error {
	return nil
}

// Apply for the credentials option is a no-op.
func (s *CredentialsOption) Apply(interface{}) error {
	return nil
}
