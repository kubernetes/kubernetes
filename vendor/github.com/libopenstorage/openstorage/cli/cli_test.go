package cli

import (
	"testing"

	"github.com/libopenstorage/openstorage/api"
	"github.com/stretchr/testify/require"
)

func TestCmdMarshalProto(t *testing.T) {
	volumeSpec := &api.VolumeSpec{
		Size:   64,
		Format: api.FSType_FS_TYPE_EXT4,
	}
	data := cmdMarshalProto(volumeSpec, false)
	require.Equal(
		t,
		`{
 "ephemeral": false,
 "size": "64",
 "format": "ext4",
 "block_size": "0",
 "ha_level": "0",
 "cos": "none",
 "io_profile": "sequential",
 "dedupe": false,
 "snapshot_interval": 0,
 "shared": false,
 "aggregation_level": 0,
 "encrypted": false,
 "passphrase": "",
 "snapshot_schedule": "",
 "scale": 0,
 "sticky": false,
 "group_enforced": false,
 "compressed": false
}`,
		data,
	)
}
