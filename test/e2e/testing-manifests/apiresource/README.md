# API Resource List and YAML Files

This directory hosts the API resource list and yaml files used for [API endpoints coverage E2E test](https://github.com/kubernetes/kubernetes/blob/master/test/e2e/apimachinery/coverage.go). The goal is to verify that a Kubernetes cluster exposes all the expected
endpoints.

## Content
This directory contains:
 * a list of API endpoints served by API server: [resources\_all.csv](./resource_all.csv),
 * the whitelist of API endpoints that are not included in the API coverage test: [resources\_whitelist.csv](./resources_whitelist.csv),
 * the actual list of API endpoints that are used in the E2E test: [resources.csv](./resources.csv).
 * the objects YAML needed to operate create/update/delete verbs in [yamlfiles/](./yamlfiles)

The csv files are in the format of:
 * **GROUP,VERSION,RESOURCE,NAMESPACED,VERB**

where **GROUP** is empty string "" for core group. **RESOURCE** may contain "/" to indicate
a subresource. **NAMESPACED** is either "*true*" or "*false*".

The YAML files for different API resources are organized in
 * **yamlfiles/\<GROUP\>/\<VERSION\>/\<RESOURCE\>.yaml**

## Test

The API coverage E2E test tries each operations in the order of
  *list, create, get, update, watch, patch, delete, deletecollection*
This test skips the verbs that doesn't appear in the API resource list.
(**NOTE**: the test uses same yaml file for create and update, and it uses an empty
JSON object to test the patch endpoint)

This test doesn't verify the correct behavior of API resources, which means the
test doesn't depends on the content of the YAML files, as long as the YAML passes
the API endpoints' validation. However, the test does expect to see no error when
operating the verbs. This test uses dynamic client for simplicity.

## Update API Resource Lists
The API resource lists are verified by hack/verify-api-resource-list.sh to
ensure the lists are up-to-date. If some API development is in progress and API
resource lists change is detected, a developer should run
hack/update-api-resource-list.sh to update the csv files. If some API resource
endpoints are added, please refer to the [Update YAML Files](#update-yaml-files) section below to
properly pass the API coverage E2E test.

**NOTE for developers and reviewers**: the entries in whitelist
([resources\_whitelist.csv](./resources_whitelist.csv)) are not expected to be added, instead we should try
remove entries from whitelist to expand the API endpoint coverage. Ideally we
should have no endpoint in the whitelist.

Currently whitelisted API endpoints include:
 * Unnamespaced resources: modifying unnamespaced API resource in parallel may
   break other e2e tests;
 * API resources that have complex dependency: e.g. creating resource A requires
   resource B created (and A is not a subresource of B). The current E2E test
   doesn't support this case;
 * Resources that we haven't added YAML files yet: contribution is welcome to
   help us expand test coverage.

## Update YAML Files
If the [API resource list](./resources.csv) gets changed AND/OR if you are making API change that adds/updates some APIs in **\<GROUP\>/\<VERSION\>/\<RESOURCE\>**, please update the corresponding yamlfiles in **test/e2e/testing-manifests/apiresource/yamlfiles/\<GROUP\>/\<VERSION\>/\<RESOURCE\>.yaml** to properly pass the API coverage e2e test (test/e2e/apimachinery/coverage.go). **NOTE**: please use plural name for resource name.

For example, if you added a new API resource **Foo** (*plural name: foos*) under API group: *core*, version:
*v2beta1*, with supported verbs: *get, update, list, create, delete*. Please add
YAML file: **yamlfiles/core/v2beta1/foos.yaml** to pass the test in order of:

  *list, create, get, update, delete*
