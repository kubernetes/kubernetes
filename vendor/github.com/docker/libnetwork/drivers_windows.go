package libnetwork

import (
	"github.com/docker/libnetwork/drivers/null"
	"github.com/docker/libnetwork/drivers/remote"
	"github.com/docker/libnetwork/drivers/windows"
	"github.com/docker/libnetwork/drivers/windows/overlay"
)

func getInitializers(experimental bool) []initializer {
	return []initializer{
		{null.Init, "null"},
		{overlay.Init, "overlay"},
		{remote.Init, "remote"},
		{windows.GetInit("transparent"), "transparent"},
		{windows.GetInit("l2bridge"), "l2bridge"},
		{windows.GetInit("l2tunnel"), "l2tunnel"},
		{windows.GetInit("nat"), "nat"},
		{windows.GetInit("ics"), "ics"},
	}
}
