Step 3: Code Hunting
====================

If you plan to submit a feature or bug fix to Gophercloud, you must be
able to prove your code correctly works with the OpenStack service in
question.

Let's use the following issue as an example:
[https://github.com/gophercloud/gophercloud/issues/621](https://github.com/gophercloud/gophercloud/issues/621).
In this issue, there's a request being made to add support for
`availability_zone_hints` to the `networking/v2/networks` package.
Meaning, we want to change:

```go
type Network struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	AdminStateUp bool     `json:"admin_state_up"`
	Status       string   `json:"status"`
	Subnets      []string `json:"subnets"`
	TenantID     string   `json:"tenant_id"`
	Shared       bool     `json:"shared"`
}
```

to look like

```go
type Network struct {
	ID                    string   `json:"id"`
	Name                  string   `json:"name"`
	AdminStateUp          bool     `json:"admin_state_up"`
	Status                string   `json:"status"`
	Subnets               []string `json:"subnets"`
	TenantID              string   `json:"tenant_id"`
	Shared                bool     `json:"shared"`

	AvailabilityZoneHints []string `json:"availability_zone_hints"`
}
```

We need to be sure that `availability_zone_hints` is a field which really does
exist in the OpenStack Neutron project and it's not a field which was added as
a customization to a single OpenStack cloud.

In addition, we need to ensure that `availability_zone_hints` is really a
`[]string` and not a different kind of type.

One way of verifying this is through the [OpenStack API reference
documentation](https://developer.openstack.org/api-ref/network/v2/).
However, the API docs might either be incorrect or they might not provide all of
the details we need to know in order to ensure this field is added correctly.

> Note: when we say the API docs might be incorrect, we are _not_ implying
> that the API docs aren't useful or that the contributors who work on the API
> docs are wrong. OpenStack moves fast. Typos happen. Forgetting to update
> documentation happens.

Since the OpenStack service itself correctly accepts and processes the fields,
the best source of information on how the field works is in the service code
itself.

Continuing on with using #621 as an example, we can find the definition of
`availability_zone_hints` in the following piece of code:

https://github.com/openstack/neutron/blob/8e9959725eda4063a318b4ba6af1e3494cad9e35/neutron/objects/network.py#L191

The above code confirms that `availability_zone_hints` is indeed part of the
`Network` object and that its type is a list of strings (`[]string`).

This example is a best-case situation: the code is relatively easy to find
and it's simple to understand. However, there will be times when proving the
implementation in the service code is difficult. Make no mistake, this is _not_
fun work. This can sometimes be more difficult than writing the actual patch
for Gophercloud. However, this is an essential step to ensuring the feature
or bug fix is correctly added to Gophercloud.

Examples of good code hunting can be seen here:

* https://github.com/gophercloud/gophercloud/issues/539
* https://github.com/gophercloud/gophercloud/issues/555
* https://github.com/gophercloud/gophercloud/issues/571
* https://github.com/gophercloud/gophercloud/issues/583
* https://github.com/gophercloud/gophercloud/issues/605

Code Hunting Tips
-----------------

OpenStack projects differ from one to another. Code is organized in different
ways. However, the following tips should be useful across all projects.

* The logic which implements Create and Delete actions is usually either located
  in the "model" or "controller" portion of the code.

* Use Github's search box to search for the exact field you're working on.
  Review all results to gain a good understanding of everywhere the field is
  used.

* When adding a field, look for an object model or a schema of some sort.

---

Proceed to [Step 4](step-04-acceptance-testing.md) to learn about Acceptance
Testing.
