package client

const (
	LOAD_BALANCER_APP_COOKIE_STICKINESS_POLICY_TYPE = "loadBalancerAppCookieStickinessPolicy"
)

type LoadBalancerAppCookieStickinessPolicy struct {
	Resource

	Cookie string `json:"cookie,omitempty" yaml:"cookie,omitempty"`

	MaxLength int64 `json:"maxLength,omitempty" yaml:"max_length,omitempty"`

	Mode string `json:"mode,omitempty" yaml:"mode,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Prefix bool `json:"prefix,omitempty" yaml:"prefix,omitempty"`

	RequestLearn bool `json:"requestLearn,omitempty" yaml:"request_learn,omitempty"`

	Timeout int64 `json:"timeout,omitempty" yaml:"timeout,omitempty"`
}

type LoadBalancerAppCookieStickinessPolicyCollection struct {
	Collection
	Data []LoadBalancerAppCookieStickinessPolicy `json:"data,omitempty"`
}

type LoadBalancerAppCookieStickinessPolicyClient struct {
	rancherClient *RancherClient
}

type LoadBalancerAppCookieStickinessPolicyOperations interface {
	List(opts *ListOpts) (*LoadBalancerAppCookieStickinessPolicyCollection, error)
	Create(opts *LoadBalancerAppCookieStickinessPolicy) (*LoadBalancerAppCookieStickinessPolicy, error)
	Update(existing *LoadBalancerAppCookieStickinessPolicy, updates interface{}) (*LoadBalancerAppCookieStickinessPolicy, error)
	ById(id string) (*LoadBalancerAppCookieStickinessPolicy, error)
	Delete(container *LoadBalancerAppCookieStickinessPolicy) error
}

func newLoadBalancerAppCookieStickinessPolicyClient(rancherClient *RancherClient) *LoadBalancerAppCookieStickinessPolicyClient {
	return &LoadBalancerAppCookieStickinessPolicyClient{
		rancherClient: rancherClient,
	}
}

func (c *LoadBalancerAppCookieStickinessPolicyClient) Create(container *LoadBalancerAppCookieStickinessPolicy) (*LoadBalancerAppCookieStickinessPolicy, error) {
	resp := &LoadBalancerAppCookieStickinessPolicy{}
	err := c.rancherClient.doCreate(LOAD_BALANCER_APP_COOKIE_STICKINESS_POLICY_TYPE, container, resp)
	return resp, err
}

func (c *LoadBalancerAppCookieStickinessPolicyClient) Update(existing *LoadBalancerAppCookieStickinessPolicy, updates interface{}) (*LoadBalancerAppCookieStickinessPolicy, error) {
	resp := &LoadBalancerAppCookieStickinessPolicy{}
	err := c.rancherClient.doUpdate(LOAD_BALANCER_APP_COOKIE_STICKINESS_POLICY_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *LoadBalancerAppCookieStickinessPolicyClient) List(opts *ListOpts) (*LoadBalancerAppCookieStickinessPolicyCollection, error) {
	resp := &LoadBalancerAppCookieStickinessPolicyCollection{}
	err := c.rancherClient.doList(LOAD_BALANCER_APP_COOKIE_STICKINESS_POLICY_TYPE, opts, resp)
	return resp, err
}

func (c *LoadBalancerAppCookieStickinessPolicyClient) ById(id string) (*LoadBalancerAppCookieStickinessPolicy, error) {
	resp := &LoadBalancerAppCookieStickinessPolicy{}
	err := c.rancherClient.doById(LOAD_BALANCER_APP_COOKIE_STICKINESS_POLICY_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *LoadBalancerAppCookieStickinessPolicyClient) Delete(container *LoadBalancerAppCookieStickinessPolicy) error {
	return c.rancherClient.doResourceDelete(LOAD_BALANCER_APP_COOKIE_STICKINESS_POLICY_TYPE, &container.Resource)
}
