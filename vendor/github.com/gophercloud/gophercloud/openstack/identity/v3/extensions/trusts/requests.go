package trusts

import "github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"

type AuthOptsExt struct {
	tokens.AuthOptionsBuilder
	TrustID string `json:"id"`
}

func (opts AuthOptsExt) ToTokenV3CreateMap(scope map[string]interface{}) (map[string]interface{}, error) {
	return opts.AuthOptionsBuilder.ToTokenV3CreateMap(scope)
}

func (opts AuthOptsExt) ToTokenV3ScopeMap() (map[string]interface{}, error) {
	b, err := opts.AuthOptionsBuilder.ToTokenV3ScopeMap()
	if err != nil {
		return nil, err
	}

	if opts.TrustID != "" {
		if b == nil {
			b = make(map[string]interface{})
		}
		b["OS-TRUST:trust"] = map[string]interface{}{
			"id": opts.TrustID,
		}
	}

	return b, nil
}

func (opts AuthOptsExt) CanReauth() bool {
	return opts.AuthOptionsBuilder.CanReauth()
}
