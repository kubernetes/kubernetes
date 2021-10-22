package v1

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/actions"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/clusters"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/nodes"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/policies"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/profiles"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/receivers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

var TestPolicySpec = policies.Spec{
	Description: "new policy description",
	Properties: map[string]interface{}{
		"destroy_after_deletion":  true,
		"grace_period":            60,
		"reduce_desired_capacity": false,
		"criteria":                "OLDEST_FIRST",
	},
	Type:    "senlin.policy.deletion",
	Version: "1.1",
}

// CreateCluster creates a random cluster. An error will be returned if
// the cluster could not be created.
func CreateCluster(t *testing.T, client *gophercloud.ServiceClient, profileID string) (*clusters.Cluster, error) {
	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create cluster: %s", name)

	createOpts := clusters.CreateOpts{
		Name:            name,
		DesiredCapacity: 1,
		ProfileID:       profileID,
		MinSize:         new(int),
		MaxSize:         20,
		Timeout:         3600,
		Metadata: map[string]interface{}{
			"foo": "bar",
			"test": map[string]interface{}{
				"nil_interface": interface{}(nil),
				"float_value":   float64(123.3),
				"string_value":  "test_string",
				"bool_value":    false,
			},
		},
		Config: map[string]interface{}{},
	}

	res := clusters.Create(client, createOpts)
	if res.Err != nil {
		return nil, res.Err
	}

	requestID := res.Header.Get("X-OpenStack-Request-Id")
	th.AssertEquals(t, true, requestID != "")
	t.Logf("Cluster %s request ID: %s", name, requestID)

	actionID, err := GetActionID(res.Header)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, actionID != "")
	t.Logf("Cluster %s action ID: %s", name, actionID)

	err = WaitForAction(client, actionID)
	if err != nil {
		return nil, err
	}

	cluster, err := res.Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created cluster: %s", cluster.ID)

	tools.PrintResource(t, cluster)
	tools.PrintResource(t, cluster.CreatedAt)

	th.AssertEquals(t, name, cluster.Name)
	th.AssertEquals(t, profileID, cluster.ProfileID)

	return cluster, nil
}

// CreateNode creates a random node. An error will be returned if
// the node could not be created.
func CreateNode(t *testing.T, client *gophercloud.ServiceClient, clusterID, profileID string) (*nodes.Node, error) {
	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create node: %s", name)

	createOpts := nodes.CreateOpts{
		ClusterID: clusterID,
		Metadata: map[string]interface{}{
			"foo": "bar",
			"test": map[string]interface{}{
				"nil_interface": interface{}(nil),
				"float_value":   float64(123.3),
				"string_value":  "test_string",
				"bool_value":    false,
			},
		},
		Name:      name,
		ProfileID: profileID,
		Role:      "",
	}

	res := nodes.Create(client, createOpts)
	if res.Err != nil {
		return nil, res.Err
	}

	requestID := res.Header.Get("X-OpenStack-Request-Id")
	th.AssertEquals(t, true, requestID != "")
	t.Logf("Node %s request ID: %s", name, requestID)

	actionID, err := GetActionID(res.Header)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, actionID != "")
	t.Logf("Node %s action ID: %s", name, actionID)

	err = WaitForAction(client, actionID)
	if err != nil {
		return nil, err
	}

	node, err := res.Extract()
	if err != nil {
		return nil, err
	}

	err = WaitForNodeStatus(client, node.ID, "ACTIVE")
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created node: %s", node.ID)

	node, err = nodes.Get(client, node.ID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, node)
	tools.PrintResource(t, node.CreatedAt)

	th.AssertEquals(t, profileID, node.ProfileID)
	th.AssertEquals(t, clusterID, node.ClusterID)
	th.AssertDeepEquals(t, createOpts.Metadata, node.Metadata)

	return node, nil
}

// CreatePolicy creates a random policy. An error will be returned if the
// policy could not be created.
func CreatePolicy(t *testing.T, client *gophercloud.ServiceClient) (*policies.Policy, error) {
	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create policy: %s", name)

	createOpts := policies.CreateOpts{
		Name: name,
		Spec: TestPolicySpec,
	}

	res := policies.Create(client, createOpts)
	if res.Err != nil {
		return nil, res.Err
	}

	requestID := res.Header.Get("X-OpenStack-Request-Id")
	th.AssertEquals(t, true, requestID != "")

	t.Logf("Policy %s request ID: %s", name, requestID)

	policy, err := res.Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created policy: %s", policy.ID)

	tools.PrintResource(t, policy)
	tools.PrintResource(t, policy.CreatedAt)

	th.AssertEquals(t, name, policy.Name)

	return policy, nil
}

// CreateProfile will create a random profile. An error will be returned if the
// profile could not be created.
func CreateProfile(t *testing.T, client *gophercloud.ServiceClient) (*profiles.Profile, error) {
	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		return nil, err
	}

	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create profile: %s", name)

	networks := []map[string]interface{}{
		{"network": choices.NetworkName},
	}

	props := map[string]interface{}{
		"name":            name,
		"flavor":          choices.FlavorID,
		"image":           choices.ImageID,
		"networks":        networks,
		"security_groups": "",
	}

	createOpts := profiles.CreateOpts{
		Name: name,
		Spec: profiles.Spec{
			Type:       "os.nova.server",
			Version:    "1.0",
			Properties: props,
		},
	}

	res := profiles.Create(client, createOpts)
	if res.Err != nil {
		return nil, res.Err
	}

	requestID := res.Header.Get("X-OpenStack-Request-Id")
	th.AssertEquals(t, true, requestID != "")

	t.Logf("Profile %s request ID: %s", name, requestID)

	profile, err := res.Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created profile: %s", profile.ID)

	tools.PrintResource(t, profile)
	tools.PrintResource(t, profile.CreatedAt)

	th.AssertEquals(t, name, profile.Name)
	th.AssertEquals(t, profile.Spec.Type, "os.nova.server")
	th.AssertEquals(t, profile.Spec.Version, "1.0")

	return profile, nil
}

// CreateWebhookReceiver will create a random webhook receiver. An error will be returned if the
// receiver could not be created.
func CreateWebhookReceiver(t *testing.T, client *gophercloud.ServiceClient, clusterID string) (*receivers.Receiver, error) {
	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create receiver: %s", name)

	createOpts := receivers.CreateOpts{
		Name:      name,
		ClusterID: clusterID,
		Type:      receivers.WebhookReceiver,
		Action:    "CLUSTER_SCALE_OUT",
	}

	res := receivers.Create(client, createOpts)
	if res.Err != nil {
		return nil, res.Err
	}

	receiver, err := res.Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created webhook receiver: %s", receiver.ID)

	tools.PrintResource(t, receiver)
	tools.PrintResource(t, receiver.CreatedAt)

	th.AssertEquals(t, name, receiver.Name)
	th.AssertEquals(t, createOpts.Action, receiver.Action)

	return receiver, nil
}

// CreateMessageReceiver will create a message receiver with a random name. An error will be returned if the
// receiver could not be created.
func CreateMessageReceiver(t *testing.T, client *gophercloud.ServiceClient, clusterID string) (*receivers.Receiver, error) {
	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create receiver: %s", name)

	createOpts := receivers.CreateOpts{
		Name:      name,
		ClusterID: clusterID,
		Type:      receivers.MessageReceiver,
	}

	res := receivers.Create(client, createOpts)
	if res.Err != nil {
		return nil, res.Err
	}

	receiver, err := res.Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created message receiver: %s", receiver.ID)

	tools.PrintResource(t, receiver)
	tools.PrintResource(t, receiver.CreatedAt)

	th.AssertEquals(t, name, receiver.Name)
	th.AssertEquals(t, createOpts.Action, receiver.Action)

	return receiver, nil
}

// DeleteCluster will delete a given policy. A fatal error will occur if the
// cluster could not be deleted. This works best as a deferred function.
func DeleteCluster(t *testing.T, client *gophercloud.ServiceClient, id string) {
	t.Logf("Attempting to delete cluster: %s", id)

	res := clusters.Delete(client, id)
	if res.Err != nil {
		t.Fatalf("Error deleting cluster %s: %s:", id, res.Err)
	}

	actionID, err := GetActionID(res.Header)
	if err != nil {
		t.Fatalf("Error deleting cluster %s: %s:", id, res.Err)
	}

	err = WaitForAction(client, actionID)
	if err != nil {
		t.Fatalf("Error deleting cluster %s: %s:", id, res.Err)
	}

	t.Logf("Successfully deleted cluster: %s", id)

	return
}

// DeleteNode will delete a given node. A fatal error will occur if the
// node could not be deleted. This works best as a deferred function.
func DeleteNode(t *testing.T, client *gophercloud.ServiceClient, id string) {
	t.Logf("Attempting to delete node: %s", id)

	res := nodes.Delete(client, id)
	if res.Err != nil {
		t.Fatalf("Error deleting node %s: %s:", id, res.Err)
	}

	actionID, err := GetActionID(res.Header)
	if err != nil {
		t.Fatalf("Error getting actionID %s: %s:", id, err)
	}

	err = WaitForAction(client, actionID)

	if err != nil {
		t.Fatalf("Error deleting node %s: %s", id, err)
	}

	t.Logf("Successfully deleted node: %s", id)

	return
}

// DeletePolicy will delete a given policy. A fatal error will occur if the
// policy could not be deleted. This works best as a deferred function.
func DeletePolicy(t *testing.T, client *gophercloud.ServiceClient, id string) {
	t.Logf("Attempting to delete policy: %s", id)

	err := policies.Delete(client, id).ExtractErr()
	if err != nil {
		t.Fatalf("Error deleting policy %s: %s:", id, err)
	}

	t.Logf("Successfully deleted policy: %s", id)

	return
}

// DeleteProfile will delete a given profile. A fatal error will occur if the
// profile could not be deleted. This works best as a deferred function.
func DeleteProfile(t *testing.T, client *gophercloud.ServiceClient, id string) {
	t.Logf("Attempting to delete profile: %s", id)

	err := profiles.Delete(client, id).ExtractErr()
	if err != nil {
		t.Fatalf("Error deleting profile %s: %s:", id, err)
	}

	t.Logf("Successfully deleted profile: %s", id)

	return
}

// DeleteReceiver will delete a given receiver. A fatal error will occur if the
// receiver could not be deleted. This works best as a deferred function.
func DeleteReceiver(t *testing.T, client *gophercloud.ServiceClient, id string) {
	t.Logf("Attempting to delete Receiver: %s", id)

	res := receivers.Delete(client, id)
	if res.Err != nil {
		t.Fatalf("Error deleting receiver %s: %s:", id, res.Err)
	}

	t.Logf("Successfully deleted receiver: %s", id)

	return
}

// GetActionID parses an HTTP header and returns the action ID.
func GetActionID(headers http.Header) (string, error) {
	location := headers.Get("Location")
	v := strings.Split(location, "actions/")
	if len(v) < 2 {
		return "", fmt.Errorf("unable to determine action ID")
	}

	actionID := v[1]

	return actionID, nil
}

func WaitForAction(client *gophercloud.ServiceClient, actionID string) error {
	return tools.WaitFor(func() (bool, error) {
		action, err := actions.Get(client, actionID).Extract()
		if err != nil {
			return false, err
		}

		if action.Status == "SUCCEEDED" {
			return true, nil
		}

		if action.Status == "FAILED" {
			return false, fmt.Errorf("Action %s in FAILED state", actionID)
		}

		return false, nil
	})
}

func WaitForNodeStatus(client *gophercloud.ServiceClient, id string, status string) error {
	return tools.WaitFor(func() (bool, error) {
		latest, err := nodes.Get(client, id).Extract()
		if err != nil {
			if _, ok := err.(gophercloud.ErrDefault404); ok && status == "DELETED" {
				return true, nil
			}

			return false, err
		}

		if latest.Status == status {
			return true, nil
		}

		if latest.Status == "ERROR" {
			return false, fmt.Errorf("Node %s in ERROR state", id)
		}

		return false, nil
	})
}
