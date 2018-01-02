package ipamutils

import (
	"testing"

	_ "github.com/docker/libnetwork/testutils"
)

func init() {
	InitNetworks()
}

func TestGranularPredefined(t *testing.T) {
	for _, nw := range PredefinedGranularNetworks {
		if ones, bits := nw.Mask.Size(); bits != 32 || ones != 24 {
			t.Fatalf("Unexpected size for network in granular list: %v", nw)
		}
	}

	for _, nw := range PredefinedBroadNetworks {
		if ones, bits := nw.Mask.Size(); bits != 32 || (ones != 20 && ones != 16) {
			t.Fatalf("Unexpected size for network in broad list: %v", nw)
		}
	}

}
