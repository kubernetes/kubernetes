package noauth

/*
Acceptance tests for Ironic endpoints with auth_strategy=noauth.  Specify
IRONIC_ENDPOINT environment variable.  For example:

  IRONIC_ENDPOINT="http://127.0.0.1:6385/v1" go test ./acceptance/openstack/baremetal/noauth/...
*/
