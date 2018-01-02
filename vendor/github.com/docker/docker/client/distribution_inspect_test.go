package client

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

func TestDistributionInspectUnsupported(t *testing.T) {
	client := &Client{
		version: "1.29",
		client:  &http.Client{},
	}
	_, err := client.DistributionInspect(context.Background(), "foobar:1.0", "")
	assert.EqualError(t, err, `"distribution inspect" requires API version 1.30, but the Docker daemon API version is 1.29`)
}
