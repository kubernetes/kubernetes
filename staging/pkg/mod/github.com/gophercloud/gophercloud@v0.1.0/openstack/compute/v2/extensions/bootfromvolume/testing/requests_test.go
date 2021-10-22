package testing

import (
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestBootFromNewVolume(t *testing.T) {

	actual, err := NewVolumeRequest.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, ExpectedNewVolumeRequest, actual)
}

func TestBootFromExistingVolume(t *testing.T) {
	actual, err := ExistingVolumeRequest.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, ExpectedExistingVolumeRequest, actual)
}

func TestBootFromImage(t *testing.T) {
	actual, err := ImageRequest.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, ExpectedImageRequest, actual)
}

func TestCreateMultiEphemeralOpts(t *testing.T) {
	actual, err := MultiEphemeralRequest.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, ExpectedMultiEphemeralRequest, actual)
}

func TestAttachNewVolume(t *testing.T) {
	actual, err := ImageAndNewVolumeRequest.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, ExpectedImageAndNewVolumeRequest, actual)
}

func TestAttachExistingVolume(t *testing.T) {
	actual, err := ImageAndExistingVolumeRequest.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, ExpectedImageAndExistingVolumeRequest, actual)
}
