angular.module("kubernetesApp.config", [])

.constant("ENV", {
	"/": {
		"k8sApiServer": "/api/v1",
		"k8sDataServer": "",
		"k8sDataPollMinIntervalSec": 10,
		"k8sDataPollMaxIntervalSec": 120,
		"k8sDataPollErrorThreshold": 5,
		"cAdvisorProxy": "",
		"cAdvisorPort": "4194"
	}
})

.constant("ngConstant", true)

;