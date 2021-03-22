package features

import (
	"k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
)

var (
	// owner: @jsafrane
	// beta: v1.21
	//
	// Enables the AWS EBS CSI migration for the Attach/Detach controller (ADC) only.
	ADCCSIMigrationAWS featuregate.Feature = "ADC_CSIMigrationAWS"

	// owner: @fbertina
	// beta: v1.22
	//
	// Enables the Azure Disk CSI migration for the Attach/Detach controller (ADC) only.
	ADCCSIMigrationAzureDisk featuregate.Feature = "ADC_CSIMigrationAzureDisk"

	// owner: @jsafrane
	// beta: v1.21
	//
	// Enables the Cinder CSI migration for the Attach/Detach controller (ADC) only.
	ADCCSIMigrationCinder featuregate.Feature = "ADC_CSIMigrationCinder"

	// owner: @fbertina
	// beta: v1.22
	//
	// Enables the GCE CSI migration for the Attach/Detach controller (ADC) only.
	ADCCSIMigrationGCEPD featuregate.Feature = "ADC_CSIMigrationGCEPD"
)

var ocpDefaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	ADCCSIMigrationAWS:       {Default: true, PreRelease: featuregate.Beta},
	ADCCSIMigrationAzureDisk: {Default: true, PreRelease: featuregate.Beta},
	ADCCSIMigrationCinder:    {Default: true, PreRelease: featuregate.Beta},
	ADCCSIMigrationGCEPD:     {Default: true, PreRelease: featuregate.Beta},
}

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.Add(ocpDefaultKubernetesFeatureGates))
}
