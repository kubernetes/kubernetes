package godo

// RegionsService is an interface for interfacing with the regions
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2#regions
type RegionsService interface {
	List(*ListOptions) ([]Region, *Response, error)
}

// RegionsServiceOp handles communication with the region related methods of the
// DigitalOcean API.
type RegionsServiceOp struct {
	client *Client
}

var _ RegionsService = &RegionsServiceOp{}

// Region represents a DigitalOcean Region
type Region struct {
	Slug      string   `json:"slug,omitempty"`
	Name      string   `json:"name,omitempty"`
	Sizes     []string `json:"sizes,omitempty"`
	Available bool     `json:"available,omitempty"`
	Features  []string `json:"features,omitempty"`
}

type regionsRoot struct {
	Regions []Region
	Links   *Links `json:"links"`
}

type regionRoot struct {
	Region Region
}

func (r Region) String() string {
	return Stringify(r)
}

// List all regions
func (s *RegionsServiceOp) List(opt *ListOptions) ([]Region, *Response, error) {
	path := "v2/regions"
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(regionsRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Regions, resp, err
}
