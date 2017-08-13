package godo

import "github.com/digitalocean/godo/context"

// SizesService is an interface for interfacing with the size
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2#sizes
type SizesService interface {
	List(context.Context, *ListOptions) ([]Size, *Response, error)
}

// SizesServiceOp handles communication with the size related methods of the
// DigitalOcean API.
type SizesServiceOp struct {
	client *Client
}

var _ SizesService = &SizesServiceOp{}

// Size represents a DigitalOcean Size
type Size struct {
	Slug         string   `json:"slug,omitempty"`
	Memory       int      `json:"memory,omitempty"`
	Vcpus        int      `json:"vcpus,omitempty"`
	Disk         int      `json:"disk,omitempty"`
	PriceMonthly float64  `json:"price_monthly,omitempty"`
	PriceHourly  float64  `json:"price_hourly,omitempty"`
	Regions      []string `json:"regions,omitempty"`
	Available    bool     `json:"available,omitempty"`
	Transfer     float64  `json:"transfer,omitempty"`
}

func (s Size) String() string {
	return Stringify(s)
}

type sizesRoot struct {
	Sizes []Size
	Links *Links `json:"links"`
}

// List all images
func (s *SizesServiceOp) List(ctx context.Context, opt *ListOptions) ([]Size, *Response, error) {
	path := "v2/sizes"
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(sizesRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Sizes, resp, err
}
