// This is a generated file. Do not edit directly.

module k8s.io/client-go

go 1.12

require (
	cloud.google.com/go v0.38.0 // indirect
	github.com/Azure/go-autorest/autorest v0.9.0
	github.com/Azure/go-autorest/autorest/adal v0.5.0
	github.com/davecgh/go-spew v1.1.1
	github.com/evanphx/json-patch v0.0.0-20200808040245-162e5629780b
	github.com/gogo/protobuf v1.2.2-0.20190723190241-65acae22fc9d
	github.com/golang/groupcache v0.0.0-20160516000752-02826c3e7903
	github.com/golang/protobuf v1.3.1
	github.com/google/gofuzz v1.0.0
	github.com/googleapis/gnostic v0.0.0-20170729233727-0c5108395e2d
	github.com/gophercloud/gophercloud v0.1.0
	github.com/gregjones/httpcache v0.0.0-20180305231024-9cad4c3443a7
	github.com/imdario/mergo v0.3.5
	github.com/peterbourgon/diskv v2.0.1+incompatible
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.3.0
	golang.org/x/crypto v0.0.0-20200220183623-bac4c82f6975
	golang.org/x/net v0.0.0-20191004110552-13f9640d40b9
	golang.org/x/oauth2 v0.0.0-20190604053449-0f29369cfe45
	golang.org/x/time v0.0.0-20190308202827-9d24e82272b4
	google.golang.org/appengine v1.5.0 // indirect
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/klog v1.0.0
	k8s.io/utils v0.0.0-20190801114015-581e00157fb1
	sigs.k8s.io/yaml v1.1.0
)

replace (
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
)
