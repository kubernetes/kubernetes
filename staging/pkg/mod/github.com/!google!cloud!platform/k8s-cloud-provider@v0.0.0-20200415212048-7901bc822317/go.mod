module github.com/GoogleCloudPlatform/k8s-cloud-provider

go 1.13

require (
	cloud.google.com/go v0.51.0 // indirect
	golang.org/x/net v0.0.0-20200114155413-6afb5195e5aa // indirect
	golang.org/x/oauth2 v0.0.0-20191202225959-858c2ad4c8b6
	golang.org/x/sys v0.0.0-20200116001909-b77594299b42 // indirect
	google.golang.org/api v0.15.1-0.20200106000736-b8fc810ca6b5
	google.golang.org/genproto v0.0.0-20200115191322-ca5a22157cba // indirect
	k8s.io/klog/v2 v2.0.0
)

replace (
	cloud.google.com/go => cloud.google.com/go v0.51.0
	golang.org/x/net => golang.org/x/net v0.0.0-20200114155413-6afb5195e5aa
	golang.org/x/oauth2 => golang.org/x/oauth2 v0.0.0-20191202225959-858c2ad4c8b6
	golang.org/x/sys => golang.org/x/sys v0.0.0-20200116001909-b77594299b42
	google.golang.org/api => google.golang.org/api v0.15.1-0.20200106000736-b8fc810ca6b5
	google.golang.org/genproto => google.golang.org/genproto v0.0.0-20200115191322-ca5a22157cba
)
