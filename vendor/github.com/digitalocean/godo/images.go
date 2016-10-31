package godo

import "fmt"

const imageBasePath = "v2/images"

// ImagesService is an interface for interfacing with the images
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2#images
type ImagesService interface {
	List(*ListOptions) ([]Image, *Response, error)
	ListDistribution(opt *ListOptions) ([]Image, *Response, error)
	ListApplication(opt *ListOptions) ([]Image, *Response, error)
	ListUser(opt *ListOptions) ([]Image, *Response, error)
	GetByID(int) (*Image, *Response, error)
	GetBySlug(string) (*Image, *Response, error)
	Update(int, *ImageUpdateRequest) (*Image, *Response, error)
	Delete(int) (*Response, error)
}

// ImagesServiceOp handles communication with the image related methods of the
// DigitalOcean API.
type ImagesServiceOp struct {
	client *Client
}

var _ ImagesService = &ImagesServiceOp{}

// Image represents a DigitalOcean Image
type Image struct {
	ID           int      `json:"id,float64,omitempty"`
	Name         string   `json:"name,omitempty"`
	Type         string   `json:"type,omitempty"`
	Distribution string   `json:"distribution,omitempty"`
	Slug         string   `json:"slug,omitempty"`
	Public       bool     `json:"public,omitempty"`
	Regions      []string `json:"regions,omitempty"`
	MinDiskSize  int      `json:"min_disk_size,omitempty"`
	Created      string   `json:"created_at,omitempty"`
}

// ImageUpdateRequest represents a request to update an image.
type ImageUpdateRequest struct {
	Name string `json:"name"`
}

type imageRoot struct {
	Image Image
}

type imagesRoot struct {
	Images []Image
	Links  *Links `json:"links"`
}

type listImageOptions struct {
	Private bool   `url:"private,omitempty"`
	Type    string `url:"type,omitempty"`
}

func (i Image) String() string {
	return Stringify(i)
}

// List lists all the images available.
func (s *ImagesServiceOp) List(opt *ListOptions) ([]Image, *Response, error) {
	return s.list(opt, nil)
}

// ListDistribution lists all the distribution images.
func (s *ImagesServiceOp) ListDistribution(opt *ListOptions) ([]Image, *Response, error) {
	listOpt := listImageOptions{Type: "distribution"}
	return s.list(opt, &listOpt)
}

// ListApplication lists all the application images.
func (s *ImagesServiceOp) ListApplication(opt *ListOptions) ([]Image, *Response, error) {
	listOpt := listImageOptions{Type: "application"}
	return s.list(opt, &listOpt)
}

// ListUser lists all the user images.
func (s *ImagesServiceOp) ListUser(opt *ListOptions) ([]Image, *Response, error) {
	listOpt := listImageOptions{Private: true}
	return s.list(opt, &listOpt)
}

// GetByID retrieves an image by id.
func (s *ImagesServiceOp) GetByID(imageID int) (*Image, *Response, error) {
	if imageID < 1 {
		return nil, nil, NewArgError("imageID", "cannot be less than 1")
	}

	return s.get(interface{}(imageID))
}

// GetBySlug retrieves an image by slug.
func (s *ImagesServiceOp) GetBySlug(slug string) (*Image, *Response, error) {
	if len(slug) < 1 {
		return nil, nil, NewArgError("slug", "cannot be blank")
	}

	return s.get(interface{}(slug))
}

// Update an image name.
func (s *ImagesServiceOp) Update(imageID int, updateRequest *ImageUpdateRequest) (*Image, *Response, error) {
	if imageID < 1 {
		return nil, nil, NewArgError("imageID", "cannot be less than 1")
	}

	if updateRequest == nil {
		return nil, nil, NewArgError("updateRequest", "cannot be nil")
	}

	path := fmt.Sprintf("%s/%d", imageBasePath, imageID)
	req, err := s.client.NewRequest("PUT", path, updateRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(imageRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.Image, resp, err
}

// Delete an image.
func (s *ImagesServiceOp) Delete(imageID int) (*Response, error) {
	if imageID < 1 {
		return nil, NewArgError("imageID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d", imageBasePath, imageID)

	req, err := s.client.NewRequest("DELETE", path, nil)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(req, nil)

	return resp, err
}

// Helper method for getting an individual image
func (s *ImagesServiceOp) get(ID interface{}) (*Image, *Response, error) {
	path := fmt.Sprintf("%s/%v", imageBasePath, ID)

	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(imageRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.Image, resp, err
}

// Helper method for listing images
func (s *ImagesServiceOp) list(opt *ListOptions, listOpt *listImageOptions) ([]Image, *Response, error) {
	path := imageBasePath
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}
	path, err = addOptions(path, listOpt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(imagesRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Images, resp, err
}
