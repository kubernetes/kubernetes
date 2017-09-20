package hcsshim

// This file contains utility functions to support storage (graph) related
// functionality.

import (
	"path/filepath"
	"syscall"

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
type DriverInfo struct {
	Flavour int
	HomeDir string
}

type driverInfo struct {
	Flavour  int
	HomeDirp *uint16
}

func convertDriverInfo(info DriverInfo) (driverInfo, error) {
	homedirp, err := syscall.UTF16PtrFromString(info.HomeDir)
	if err != nil {
		logrus.Debugf("Failed conversion of home to pointer for driver info: %s", err.Error())
		return driverInfo{}, err
	}

	return driverInfo{
		Flavour:  info.Flavour,
		HomeDirp: homedirp,
	}, nil
}

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
	LayerId GUID
	Flags   uint32
	Pathp   *uint16
}

func layerPathsToDescriptors(parentLayerPaths []string) ([]WC_LAYER_DESCRIPTOR, error) {
	// Array of descriptors that gets constructed.
	var layers []WC_LAYER_DESCRIPTOR

	for i := 0; i < len(parentLayerPaths); i++ {
		// Create a layer descriptor, using the folder name
		// as the source for a GUID LayerId
		_, folderName := filepath.Split(parentLayerPaths[i])
		g, err := NameToGuid(folderName)
		if err != nil {
			logrus.Debugf("Failed to convert name to guid %s", err)
			return nil, err
		}

		p, err := syscall.UTF16PtrFromString(parentLayerPaths[i])
		if err != nil {
			logrus.Debugf("Failed conversion of parentLayerPath to pointer %s", err)
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
