package gpu

import (
	nvd "github.com/haniceboy/nvidia-docker/tools/src/nvidia"
)

func Detect(gpuDevice *GPUDevice) error {
	//drv, err := nvd.GetDriverVersion()
	_, err := nvd.GetDriverVersion()
	return err
}
