package api

type Node struct {
	Node            string
	Address         string
	TaggedAddresses map[string]string
}

type CatalogService struct {
	Node                     string
	Address                  string
	TaggedAddresses          map[string]string
	ServiceID                string
	ServiceName              string
	ServiceAddress           string
	ServiceTags              []string
	ServicePort              int
	ServiceEnableTagOverride bool
}

type CatalogNode struct {
	Node     *Node
	Services map[string]*AgentService
}

type CatalogRegistration struct {
	Node            string
	Address         string
	TaggedAddresses map[string]string
	Datacenter      string
	Service         *AgentService
	Check           *AgentCheck
}

type CatalogDeregistration struct {
	Node       string
	Address    string
	Datacenter string
	ServiceID  string
	CheckID    string
}

// Catalog can be used to query the Catalog endpoints
type Catalog struct {
	c *Client
}

// Catalog returns a handle to the catalog endpoints
func (c *Client) Catalog() *Catalog {
	return &Catalog{c}
}

func (c *Catalog) Register(reg *CatalogRegistration, q *WriteOptions) (*WriteMeta, error) {
	r := c.c.newRequest("PUT", "/v1/catalog/register")
	r.setWriteOptions(q)
	r.obj = reg
	rtt, resp, err := requireOK(c.c.doRequest(r))
	if err != nil {
		return nil, err
	}
	resp.Body.Close()

	wm := &WriteMeta{}
	wm.RequestTime = rtt

	return wm, nil
}

func (c *Catalog) Deregister(dereg *CatalogDeregistration, q *WriteOptions) (*WriteMeta, error) {
	r := c.c.newRequest("PUT", "/v1/catalog/deregister")
	r.setWriteOptions(q)
	r.obj = dereg
	rtt, resp, err := requireOK(c.c.doRequest(r))
	if err != nil {
		return nil, err
	}
	resp.Body.Close()

	wm := &WriteMeta{}
	wm.RequestTime = rtt

	return wm, nil
}

// Datacenters is used to query for all the known datacenters
func (c *Catalog) Datacenters() ([]string, error) {
	r := c.c.newRequest("GET", "/v1/catalog/datacenters")
	_, resp, err := requireOK(c.c.doRequest(r))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out []string
	if err := decodeBody(resp, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Nodes is used to query all the known nodes
func (c *Catalog) Nodes(q *QueryOptions) ([]*Node, *QueryMeta, error) {
	r := c.c.newRequest("GET", "/v1/catalog/nodes")
	r.setQueryOptions(q)
	rtt, resp, err := requireOK(c.c.doRequest(r))
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	qm := &QueryMeta{}
	parseQueryMeta(resp, qm)
	qm.RequestTime = rtt

	var out []*Node
	if err := decodeBody(resp, &out); err != nil {
		return nil, nil, err
	}
	return out, qm, nil
}

// Services is used to query for all known services
func (c *Catalog) Services(q *QueryOptions) (map[string][]string, *QueryMeta, error) {
	r := c.c.newRequest("GET", "/v1/catalog/services")
	r.setQueryOptions(q)
	rtt, resp, err := requireOK(c.c.doRequest(r))
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	qm := &QueryMeta{}
	parseQueryMeta(resp, qm)
	qm.RequestTime = rtt

	var out map[string][]string
	if err := decodeBody(resp, &out); err != nil {
		return nil, nil, err
	}
	return out, qm, nil
}

// Service is used to query catalog entries for a given service
func (c *Catalog) Service(service, tag string, q *QueryOptions) ([]*CatalogService, *QueryMeta, error) {
	r := c.c.newRequest("GET", "/v1/catalog/service/"+service)
	r.setQueryOptions(q)
	if tag != "" {
		r.params.Set("tag", tag)
	}
	rtt, resp, err := requireOK(c.c.doRequest(r))
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	qm := &QueryMeta{}
	parseQueryMeta(resp, qm)
	qm.RequestTime = rtt

	var out []*CatalogService
	if err := decodeBody(resp, &out); err != nil {
		return nil, nil, err
	}
	return out, qm, nil
}

// Node is used to query for service information about a single node
func (c *Catalog) Node(node string, q *QueryOptions) (*CatalogNode, *QueryMeta, error) {
	r := c.c.newRequest("GET", "/v1/catalog/node/"+node)
	r.setQueryOptions(q)
	rtt, resp, err := requireOK(c.c.doRequest(r))
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	qm := &QueryMeta{}
	parseQueryMeta(resp, qm)
	qm.RequestTime = rtt

	var out *CatalogNode
	if err := decodeBody(resp, &out); err != nil {
		return nil, nil, err
	}
	return out, qm, nil
}
