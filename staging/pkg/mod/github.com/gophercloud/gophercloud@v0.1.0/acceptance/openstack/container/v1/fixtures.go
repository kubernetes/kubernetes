package v1

const capsuleTemplate = `
	{
		"capsuleVersion": "beta",
		"kind": "capsule",
		"metadata": {
			"labels": {
				"app": "web",
				"app1": "web1"
			},
			"name": "template"
		},
		"spec": {
			"restartPolicy": "Always",
			"containers": [
				{
					"command": [
						"sleep",
						"1000000"
					],
					"env": {
						"ENV1": "/usr/local/bin",
						"ENV2": "/usr/bin"
					},
					"image": "ubuntu",
					"ports": [
						{
							"containerPort": 80,
							"hostPort": 80,
							"name": "nginx-port",
							"protocol": "TCP"
						}
					],
					"resources": {
						"requests": {
							"cpu": 1,
							"memory": 1024
						}
					},
					"workDir": "/root"
				}
			]
		}
	}
`
