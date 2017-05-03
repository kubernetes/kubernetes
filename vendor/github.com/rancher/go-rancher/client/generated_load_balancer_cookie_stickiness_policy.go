package client

const (
	LOAD_BALANCER_COOKIE_STICKINESS_POLICY_TYPE = "loadBalancerCookieStickinessPolicy"
)

type LoadBalancerCookieStickinessPolicy struct {
	Resource

	Cookie string `json:"cookie,omitempty" yaml:"cookie,omitempty"`

	Domain string `json:"domain,omitempty" yaml:"domain,omitempty"`

	Indirect bool `json:"indirect,omitempty" yaml:"indirect,omitempty"`

	Mode string `json:"mode,omitempty" yaml:"mode,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Nocache bool `json:"nocache,omitempty" yaml:"nocache,omitempty"`

	Postonly bool `json:"postonly,omitempty" yaml:"postonly,omitempty"`
}

type LoadBalancerCookieStickinessPolicyCollection struct {
	Collection
	Data []LoadBalancerCookieStickinessPolicy `json:"data,omitempty"`
}

type LoadBalancerCookieStickinessPolicyClient struct {
	rancherClient *RancherClient
}

type LoadBalancerCookieStickinessPolicyOperations interface {
	List(opts *ListOpts) (*LoadBalancerCookieStickinessPolicyCollection, error)
	Create(opts *LoadBalancerCookieStickinessPolicy) (*LoadBalancerCookieStickinessPolicy, error)
	Update(existing *LoadBalancerCookieStickinessPolicy, updates interface{}) (*LoadBalancerCookieStickinessPolicy, error)
	ById(id string) (*LoadBalancerCookieStickinessPolicy, error)
	Delete(container *LoadBalancerCookieStickinessPolicy) error
}

func newLoadBalancerCookieStickinessPolicyClient(rancherClient *RancherClient) *LoadBalancerCookieStickinessPolicyClient {
	return &LoadBalancerCookieStickinessPolicyClient{
		rancherClient: rancherClient,
	}
}

func (c *LoadBalancerCookieStickinessPolicyClient) Create(container *LoadBalancerCookieStickinessPolicy) (*LoadBalancerCookieStickinessPolicy, error) {
	resp := &LoadBalancerCookieStickinessPolicy{}
	err := c.rancherClient.doCreate(LOAD_BALANCER_COOKIE_STICKINESS_POLICY_TYPE, container, resp)
	return resp, err
}

func (c *LoadBalancerCookieStickinessPolicyClient) Update(existing *LoadBalancerCookieStickinessPolicy, updates interface{}) (*LoadBalancerCookieStickinessPolicy, error) {
	resp := &LoadBalancerCookieStickinessPolicy{}
	err := c.rancherClient.doUpdate(LOAD_BALANCER_COOKIE_STICKINESS_POLICY_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *LoadBalancerCookieStickinessPolicyClient) List(opts *ListOpts) (*LoadBalancerCookieStickinessPolicyCollection, error) {
	resp := &LoadBalancerCookieStickinessPolicyCollection{}
	err := c.rancherClient.doList(LOAD_BALANCER_COOKIE_STICKINESS_POLICY_TYPE, opts, resp)
	return resp, err
}

func (c *LoadBalancerCookieStickinessPolicyClient) ById(id string) (*LoadBalancerCookieStickinessPolicy, error) {
	resp := &LoadBalancerCookieStickinessPolicy{}
	err := c.rancherClient.doById(LOAD_BALANCER_COOKIE_STICKINESS_POLICY_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *LoadBalancerCookieStickinessPolicyClient) Delete(container *LoadBalancerCookieStickinessPolicy) error {
	return c.rancherClient.doResourceDelete(LOAD_BALANCER_COOKIE_STICKINESS_POLICY_TYPE, &container.Resource)
}
