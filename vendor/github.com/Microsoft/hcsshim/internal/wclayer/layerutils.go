package wclayer

// This file contains utility functions to support storage (graph) related
// functionality.

import (
	"context"
	"syscall"

	"github.com/Microsoft/go-winio/pkg/guid"
	"github.com/sirupsen/logrus"
)

/* To pass into syscall, we need a struct matching the following:
enum GraphDriverType
{
    DiffDriver,
    FilterDriver
};

struct DriverInfo {
    GraphDriverType Flavour;
    LPCWSTR HomeDir;
};
*/

type driverInfo struct {
	Flavour  int
	HomeDirp *uint16
}

var (
	utf16EmptyString uint16
	stdDriverInfo    = driverInfo{1, &utf16EmptyString}
)

/* To pass into syscall, we need a struct matching the following:
typedef struct _WC_LAYER_DESCRIPTOR {

    //
    // The ID of the layer
    //

    GUID LayerId;

    //
    // Additional flags
    //

    union {
        struct {
            ULONG Reserved : 31;
            ULONG Dirty : 1;    // Created from sandbox as a result of snapshot
        };
        ULONG Value;
    } Flags;

    //
    // Path to the layer root directory, null-terminated
    //

    PCWSTR Path;

} WC_LAYER_DESCRIPTOR, *PWC_LAYER_DESCRIPTOR;
*/
type WC_LAYER_DESCRIPTOR struct {
	LayerId guid.GUID
	Flags   uint32
	Pathp   *uint16
}

func layerPathsToDescriptors(ctx context.Context, parentLayerPaths []string) ([]WC_LAYER_DESCRIPTOR, error) {
	// Array of descriptors that gets constructed.
	var layers []WC_LAYER_DESCRIPTOR

	for i := 0; i < len(parentLayerPaths); i++ {
		g, err := LayerID(ctx, parentLayerPaths[i])
		if err != nil {
			logrus.WithError(err).Debug("Failed to convert name to guid")
			return nil, err
		}

		p, err := syscall.UTF16PtrFromString(parentLayerPaths[i])
		if err != nil {
			logrus.WithError(err).Debug("Failed conversion of parentLayerPath to pointer")
			return nil, err
		}

		layers = append(layers, WC_LAYER_DESCRIPTOR{
			LayerId: g,
			Flags:   0,
			Pathp:   p,
		})
	}

	return layers, nil
}
