package criu

import (
	"fmt"

	"github.com/checkpoint-restore/go-criu/v6/rpc"
)

// Feature checking in go-criu is based on the libcriu feature checking function.

// Feature checking allows the user to check if CRIU supports
// certain features. There are CRIU features which do not depend
// on the version of CRIU but on kernel features or architecture.
//
// One example is memory tracking. Memory tracking can be disabled
// in the kernel or there are architectures which do not support
// it (aarch64 for example). By using the feature check a libcriu
// user can easily query CRIU if a certain feature is available.
//
// The features which should be checked can be marked in the
// structure 'struct criu_feature_check'. Each structure member
// that is set to true will result in CRIU checking for the
// availability of that feature in the current combination of
// CRIU/kernel/architecture.
//
// Available features will be set to true when the function
// returns successfully. Missing features will be set to false.

func (c *Criu) FeatureCheck(features *rpc.CriuFeatures) (*rpc.CriuFeatures, error) {
	resp, err := c.doSwrkWithResp(
		rpc.CriuReqType_FEATURE_CHECK,
		nil,
		nil,
		features,
	)
	if err != nil {
		return nil, err
	}

	if resp.GetType() != rpc.CriuReqType_FEATURE_CHECK {
		return nil, fmt.Errorf("Unexpected CRIU RPC response")
	}

	return features, nil
}
