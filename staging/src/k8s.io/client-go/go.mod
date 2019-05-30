// This is a generated file. Do not edit directly.

module k8s.io/client-go

go 1.12

require (
	github.com/Azure/go-autorest v11.1.2+incompatible
	github.com/davecgh/go-spew v1.1.1
	github.com/dgrijalva/jwt-go v0.0.0-20160705203006-01aeca54ebda // indirect
	github.com/evanphx/json-patch v0.0.0-20190203023257-5858425f7550
	github.com/gogo/protobuf v0.0.0-20171007142547-342cbe0a0415
	github.com/golang/groupcache v0.0.0-20160516000752-02826c3e7903
	github.com/golang/protobuf v1.2.0
	github.com/google/btree v0.0.0-20160524151835-7d79101e329e // indirect
	github.com/google/gofuzz v0.0.0-20170612174753-24818f796faf
	github.com/googleapis/gnostic v0.0.0-20170729233727-0c5108395e2d
	github.com/gophercloud/gophercloud v0.0.0-20190126172459-c818fa66e4c8
	github.com/gregjones/httpcache v0.0.0-20170728041850-787624de3eb7
	github.com/imdario/mergo v0.3.5
	github.com/peterbourgon/diskv v2.0.1+incompatible
	github.com/spf13/pflag v1.0.1
	github.com/stretchr/testify v1.2.2
	golang.org/x/crypto v0.0.0-20181025213731-e84da0312774
	golang.org/x/net v0.0.0-20190206173232-65e2d4e15006
	golang.org/x/oauth2 v0.0.0-20190402181905-9f3314589c9a
	golang.org/x/time v0.0.0-20161028155119-f51c12702a4d
	google.golang.org/appengine v1.5.0 // indirect
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/klog v0.3.1
	k8s.io/utils v0.0.0-20190221042446-c2654d5206da
	sigs.k8s.io/yaml v1.1.0
)

replace (
	golang.org/x/sync => golang.org/x/sync v0.0.0-20181108010431-42b317875d0f
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190209173611-3b5209105503
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190313210603-aa82965741a9
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
)
