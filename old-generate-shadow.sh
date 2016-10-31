#!/bin/bash

# create a copy of pkg/volume, which uses versioned types.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

readonly SERVICEACCOUNT_DIR="${DIR}/serviceaccount"
readonly SERVICEACCOUNT_SHADOW_DIR="${DIR}/serviceaccountshadow"

rm -rf "${SERVICEACCOUNT_SHADOW_DIR}"
cp -r "${SERVICEACCOUNT_DIR}" "${SERVICEACCOUNT_SHADOW_DIR}"

#rename package
#find "${FIELDPATH_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,package fieldpath$,package fieldpathshadow,g"

#rename canonical path
find "${SERVICEACCOUNT_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,k8s.io/kubernetes/pkg/serviceaccount,k8s.io/kubernetes/pkg/serviceaccountshadow,g'

gofmt -w -r 'api.a -> v1.a' "${SERVICEACCOUNT_SHADOW_DIR}"
# Scheme is still in api
gofmt -w -r 'v1.Scheme -> api.Scheme' "${SERVICEACCOUNT_SHADOW_DIR}"

# still need pkg api, for api.Scheme
find "${SERVICEACCOUNT_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,"k8s.io/kubernetes/pkg/api","k8s.io/client-go/1.5/pkg/api"\n   "k8s.io/client-go/1.5/pkg/api/v1",g'

find "${SERVICEACCOUNT_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,k8s.io/kubernetes/pkg/api/meta",k8s.io/client-go/1.5/pkg/api/meta",g'

# change clientset
find "${SERVICEACCOUNT_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset,k8s.io/client-go/1.5/kubernetes,g"

# remove registryGetter
# to do it less hacky, we can split registryGetter in its own file.
sed -i '/registryGetter/,$d' "${SERVICEACCOUNT_SHADOW_DIR}/tokengetter.go"

goimports -w "${SERVICEACCOUNT_SHADOW_DIR}"

exit 0
###########################

readonly FIELDPATH_DIR="${DIR}/fieldpath"
readonly FIELDPATH_SHADOW_DIR="${DIR}/fieldpathshadow"

rm -rf "${FIELDPATH_SHADOW_DIR}"
cp -r "${FIELDPATH_DIR}" "${FIELDPATH_SHADOW_DIR}"

#rename package
#find "${FIELDPATH_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,package fieldpath$,package fieldpathshadow,g"

#rename canonical path
find "${FIELDPATH_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,k8s.io/kubernetes/pkg/fieldpath,k8s.io/kubernetes/pkg/fieldpathshadow,g'

gofmt -w -r 'api.a -> v1.a' "${FIELDPATH_SHADOW_DIR}"
# Scheme is still in api
gofmt -w -r 'v1.Scheme -> api.Scheme' "${FIELDPATH_SHADOW_DIR}"

# still need pkg api, for api.Scheme
find "${FIELDPATH_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,"k8s.io/kubernetes/pkg/api","k8s.io/client-go/1.5/pkg/api"\n   "k8s.io/client-go/1.5/pkg/api/v1",g'
find "${FIELDPATH_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,k8s.io/kubernetes/pkg/api/resource,k8s.io/client-go/1.5/pkg/api/resource,g"
find "${FIELDPATH_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,k8s.io/kubernetes/pkg/api/meta",k8s.io/client-go/1.5/pkg/api/meta",g'
###################################
readonly VOLUME_DIR="${DIR}/volume"
readonly VOLUME_SHADOW_DIR="${DIR}/volumeshadow"

rm -rf "${VOLUME_SHADOW_DIR}"
cp -r "${VOLUME_DIR}" "${VOLUME_SHADOW_DIR}"

gofmt -w -r 'api.a -> v1.a' "${VOLUME_SHADOW_DIR}"
gofmt -w -r 'volume.a -> volumeshadow.a' "${VOLUME_SHADOW_DIR}"
# the order of the next two lines is important
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,k8s.io/kubernetes/pkg/api/unversioned",k8s.io/client-go/1.5/pkg/api/unversioned",g'
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,k8s.io/kubernetes/pkg/api",k8s.io/client-go/1.5/pkg/api/v1",g'
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,package volume$,package volumeshadow,g"
sed -i "s,package volume,package volumeshadow,g" "${VOLUME_SHADOW_DIR}"/doc.go
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,package volume_test,package volumeshadow_test,g"
#rewrite the canonical import path
# covered by next rewrite
#find "${VOLUME_SHADOW_DIR}" -type f -name "doc.go" -print0 | xargs -0 sed -i 's,// import "k8s.io/kubernetes/pkg/volume,// import "k8s.io/kubernetes/pkg/volumeshadow,g'
#rewrite import of pkg/volume to pkg/volumeshadow for sub pacakges
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,k8s.io/kubernetes/pkg/volume,k8s.io/kubernetes/pkg/volumeshadow,g'

#rewrite uses of internalclientset to use client-go
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset,k8s.io/client-go/1.5/kubernetes,g"
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,internalclientset\.,kubernetes.,g"

find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,k8s.io/kubernetes/pkg/watch,k8s.io/client-go/1.5/pkg/watch,g"
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,k8s.io/kubernetes/pkg/fields,k8s.io/client-go/1.5/pkg/fields,g"
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,k8s.io/kubernetes/pkg/api/resource,k8s.io/client-go/1.5/pkg/api/resource,g"
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,k8s.io/kubernetes/pkg/client/record,k8s.io/client-go/1.5/tools/record,g"
# SPECIAL:types.NodeName is used in pkg/cloudprovider as well, so we want to keep using main repo's types.NodeName
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" ! -path "*util/types/*" -print0 | xargs -0 sed -i 's,"k8s.io/kubernetes/pkg/types",k8stypes "k8s.io/kubernetes/pkg/types"\n   "k8s.io/client-go/1.5/pkg/types",g'
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i 's,types.NodeName,k8stypes.NodeName,g'

#rewrite to use fieldpathshadow
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,k8s.io/kubernetes/pkg/fieldpath,k8s.io/kubernetes/pkg/fieldpathshadow,g"

# specific rewrite for using Selector in v1.ListOptions
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,FieldSelector: podSelector,FieldSelector: podSelector.String(),g"
find "${VOLUME_SHADOW_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,FieldSelector: eventSelector,FieldSelector: eventSelector.String(),g"

goimports -w "${VOLUME_SHADOW_DIR}"
