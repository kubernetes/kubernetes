package options

import (
	"github.com/spf13/pflag"
)


type ClusterController struct {
	Ubernetes       string
}

func NewClusterController() *ClusterController {
	c := ClusterController {
	}

	return &c
}

func (c *ClusterController) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&c.Ubernetes, "ubernetes", c.Ubernetes, "The address of the Ubernetes API server")
}