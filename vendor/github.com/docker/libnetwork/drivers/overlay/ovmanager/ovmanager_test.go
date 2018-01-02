package ovmanager

import (
	"fmt"
	"net"
	"strings"
	"testing"

	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/idm"
	"github.com/docker/libnetwork/netlabel"
	_ "github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newDriver(t *testing.T) *driver {
	d := &driver{
		networks: networkTable{},
	}

	vxlanIdm, err := idm.New(nil, "vxlan-id", vxlanIDStart, vxlanIDEnd)
	require.NoError(t, err)

	d.vxlanIdm = vxlanIdm
	return d
}

func parseCIDR(t *testing.T, ipnet string) *net.IPNet {
	subnet, err := types.ParseCIDR(ipnet)
	require.NoError(t, err)
	return subnet
}

func TestNetworkAllocateFree(t *testing.T) {
	d := newDriver(t)

	ipamData := []driverapi.IPAMData{
		{
			Pool: parseCIDR(t, "10.1.1.0/24"),
		},
		{
			Pool: parseCIDR(t, "10.1.2.0/24"),
		},
	}

	vals, err := d.NetworkAllocate("testnetwork", nil, ipamData, nil)
	require.NoError(t, err)

	vxlanIDs, ok := vals[netlabel.OverlayVxlanIDList]
	assert.Equal(t, true, ok)
	assert.Equal(t, 2, len(strings.Split(vxlanIDs, ",")))

	err = d.NetworkFree("testnetwork")
	require.NoError(t, err)
}

func TestNetworkAllocateUserDefinedVNIs(t *testing.T) {
	d := newDriver(t)

	ipamData := []driverapi.IPAMData{
		{
			Pool: parseCIDR(t, "10.1.1.0/24"),
		},
		{
			Pool: parseCIDR(t, "10.1.2.0/24"),
		},
	}

	options := make(map[string]string)
	// Intentionally add mode vnis than subnets
	options[netlabel.OverlayVxlanIDList] = fmt.Sprintf("%d,%d,%d", vxlanIDStart, vxlanIDStart+1, vxlanIDStart+2)

	vals, err := d.NetworkAllocate("testnetwork", options, ipamData, nil)
	require.NoError(t, err)

	vxlanIDs, ok := vals[netlabel.OverlayVxlanIDList]
	assert.Equal(t, true, ok)

	// We should only get exactly the same number of vnis as
	// subnets. No more, no less, even if we passed more vnis.
	assert.Equal(t, 2, len(strings.Split(vxlanIDs, ",")))
	assert.Equal(t, fmt.Sprintf("%d,%d", vxlanIDStart, vxlanIDStart+1), vxlanIDs)

	err = d.NetworkFree("testnetwork")
	require.NoError(t, err)
}
