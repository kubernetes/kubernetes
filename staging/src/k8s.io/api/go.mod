// This is a generated file. Do not edit directly.

module k8s.io/api

go 1.12

require (
	github.com/gogo/protobuf v0.0.0-20190410021324-765b5b8d2dfc
	github.com/stretchr/testify v1.2.2
	k8s.io/apimachinery v0.0.0
)

replace (
	github.com/gogo/protobuf => github.com/apelisse/protobuf v0.0.0-20190410021324-765b5b8d2dfc
	golang.org/x/sync => golang.org/x/sync v0.0.0-20181108010431-42b317875d0f
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190209173611-3b5209105503
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190313210603-aa82965741a9
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/gengo => k8s.io/gengo v0.0.0-20190116091435-f8a0810f38af
	sigs.k8s.io/structured-merge-diff => github.com/apelisse/structured-merge-diff v0.0.0-20190628201129-e230a57d7a
)
