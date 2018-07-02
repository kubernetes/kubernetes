module k8s.io/client-go

require (
	k8s.io/api v1.12.0
	k8s.io/apimachinery v1.12.0
)

replace (
	k8s.io/api v1.12.0 => ../api
	k8s.io/apimachinery v1.12.0 => ../apimachinery
)

require (
	cloud.google.com/go v0.0.0-20160913182117-3b1ae45394a2 // indirect
	github.com/Azure/go-autorest v0.0.0-20180418234723-1ff28809256a
	github.com/dgrijalva/jwt-go v0.0.0-20160705203006-01aeca54ebda // indirect
	github.com/ghodss/yaml v0.0.0-20150909031657-73d445a93680
	github.com/gogo/protobuf v0.0.0-20170330071051-c0656edd0d9e
	github.com/golang/glog v0.0.0-20141105023935-44145f04b68c
	github.com/golang/groupcache v0.0.0-20160516000752-02826c3e7903
	github.com/golang/protobuf v1.1.0
	github.com/google/btree v0.0.0-20160524151835-7d79101e329e // indirect
	github.com/google/gofuzz v0.0.0-20161122191042-44d81051d367
	github.com/googleapis/gnostic v0.0.0-20170729233727-0c5108395e2d
	github.com/gophercloud/gophercloud v0.0.0-20180330165814-781450b3c4fc
	github.com/gregjones/httpcache v0.0.0-20170728041850-787624de3eb7
	github.com/imdario/mergo v0.0.0-20141206190957-6633656539c1
	github.com/peterbourgon/diskv v0.0.0-20170814173558-5f041e8faa00
	github.com/spf13/pflag v1.0.1
	github.com/stretchr/testify v0.0.0-20180319223459-c679ae2cc0cb
	golang.org/x/crypto v0.0.0-20180222182404-49796115aa4b
	golang.org/x/net v0.0.0-20170809000501-1c05540f6879
	golang.org/x/oauth2 v0.0.0-20170412232759-a6bd8cefa181
	golang.org/x/time v0.0.0-20161028155119-f51c12702a4d
	google.golang.org/appengine v1.1.0 // indirect
	gopkg.in/yaml.v1 v1.0.0-20140924161607-9f9df34309c0 // indirect
)
