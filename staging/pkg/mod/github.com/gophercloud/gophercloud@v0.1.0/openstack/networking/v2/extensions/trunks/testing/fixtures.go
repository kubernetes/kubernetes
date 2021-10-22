package testing

import (
	"time"

	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/trunks"
)

const CreateRequest = `
{
  "trunk": {
    "admin_state_up": true,
    "description": "Trunk created by gophercloud",
    "name": "gophertrunk",
    "port_id": "c373d2fa-3d3b-4492-924c-aff54dea19b6",
    "sub_ports": [
      {
        "port_id": "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
        "segmentation_id": 1,
        "segmentation_type": "vlan"
      },
      {
        "port_id": "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
        "segmentation_id": 2,
        "segmentation_type": "vlan"
      }
    ]
  }
}`

const CreateResponse = `
{
  "trunk": {
    "admin_state_up": true,
    "created_at": "2018-10-03T13:57:24Z",
    "description": "Trunk created by gophercloud",
    "id": "f6a9718c-5a64-43e3-944f-4deccad8e78c",
    "name": "gophertrunk",
    "port_id": "c373d2fa-3d3b-4492-924c-aff54dea19b6",
    "project_id": "e153f3f9082240a5974f667cfe1036e3",
    "revision_number": 1,
    "status": "ACTIVE",
    "sub_ports": [
      {
        "port_id": "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
        "segmentation_id": 1,
        "segmentation_type": "vlan"
      },
      {
        "port_id": "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
        "segmentation_id": 2,
        "segmentation_type": "vlan"
      }
    ],
    "tags": [],
    "tenant_id": "e153f3f9082240a5974f667cfe1036e3",
    "updated_at": "2018-10-03T13:57:26Z"
  }
}`

const CreateNoSubportsRequest = `
{
  "trunk": {
    "admin_state_up": true,
    "description": "Trunk created by gophercloud",
    "name": "gophertrunk",
    "port_id": "c373d2fa-3d3b-4492-924c-aff54dea19b6",
    "sub_ports": []
  }
}`

const CreateNoSubportsResponse = `
{
  "trunk": {
    "admin_state_up": true,
    "created_at": "2018-10-03T13:57:24Z",
    "description": "Trunk created by gophercloud",
    "id": "f6a9718c-5a64-43e3-944f-4deccad8e78c",
    "name": "gophertrunk",
    "port_id": "c373d2fa-3d3b-4492-924c-aff54dea19b6",
    "project_id": "e153f3f9082240a5974f667cfe1036e3",
    "revision_number": 1,
    "status": "ACTIVE",
    "sub_ports": [],
    "tags": [],
    "tenant_id": "e153f3f9082240a5974f667cfe1036e3",
    "updated_at": "2018-10-03T13:57:26Z"
  }
}`

const ListResponse = `
{
  "trunks": [
    {
      "admin_state_up": true,
      "created_at": "2018-10-01T15:29:39Z",
      "description": "",
      "id": "3e72aa1b-d0da-48f2-831a-fd1c5f3f99c2",
      "name": "mytrunk",
      "port_id": "16c425d3-d7fc-40b8-b94c-cc95da45b270",
      "project_id": "e153f3f9082240a5974f667cfe1036e3",
      "revision_number": 3,
      "status": "ACTIVE",
      "sub_ports": [
        {
          "port_id": "424da4b7-7868-4db2-bb71-05155601c6e4",
          "segmentation_id": 11,
          "segmentation_type": "vlan"
        }
      ],
      "tags": [],
      "tenant_id": "e153f3f9082240a5974f667cfe1036e3",
      "updated_at": "2018-10-01T15:43:04Z"
    },
    {
      "admin_state_up": true,
      "created_at": "2018-10-03T13:57:24Z",
      "description": "Trunk created by gophercloud",
      "id": "f6a9718c-5a64-43e3-944f-4deccad8e78c",
      "name": "gophertrunk",
      "port_id": "c373d2fa-3d3b-4492-924c-aff54dea19b6",
      "project_id": "e153f3f9082240a5974f667cfe1036e3",
      "revision_number": 1,
      "status": "ACTIVE",
      "sub_ports": [
        {
          "port_id": "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
          "segmentation_id": 1,
          "segmentation_type": "vlan"
        },
        {
          "port_id": "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
          "segmentation_id": 2,
          "segmentation_type": "vlan"
        }
      ],
      "tags": [],
      "tenant_id": "e153f3f9082240a5974f667cfe1036e3",
      "updated_at": "2018-10-03T13:57:26Z"
    }
  ]
}`

const GetResponse = `
{
  "trunk": {
    "admin_state_up": true,
    "created_at": "2018-10-03T13:57:24Z",
    "description": "Trunk created by gophercloud",
    "id": "f6a9718c-5a64-43e3-944f-4deccad8e78c",
    "name": "gophertrunk",
    "port_id": "c373d2fa-3d3b-4492-924c-aff54dea19b6",
    "project_id": "e153f3f9082240a5974f667cfe1036e3",
    "revision_number": 1,
    "status": "ACTIVE",
    "sub_ports": [
      {
        "port_id": "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
        "segmentation_id": 1,
        "segmentation_type": "vlan"
      },
      {
        "port_id": "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
        "segmentation_id": 2,
        "segmentation_type": "vlan"
      }
    ],
    "tags": [],
    "tenant_id": "e153f3f9082240a5974f667cfe1036e3",
    "updated_at": "2018-10-03T13:57:26Z"
  }
}`

const UpdateRequest = `
{
  "trunk": {
    "admin_state_up": false,
    "description": "gophertrunk updated by gophercloud",
    "name": "updated_gophertrunk"
  }
}`

const UpdateResponse = `
{
  "trunk": {
    "admin_state_up": false,
    "created_at": "2018-10-03T13:57:24Z",
    "description": "gophertrunk updated by gophercloud",
    "id": "f6a9718c-5a64-43e3-944f-4deccad8e78c",
    "name": "updated_gophertrunk",
    "port_id": "c373d2fa-3d3b-4492-924c-aff54dea19b6",
    "project_id": "e153f3f9082240a5974f667cfe1036e3",
    "revision_number": 6,
    "status": "ACTIVE",
    "sub_ports": [
      {
        "port_id": "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
        "segmentation_id": 1,
        "segmentation_type": "vlan"
      },
      {
        "port_id": "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
        "segmentation_id": 2,
        "segmentation_type": "vlan"
      }
    ],
    "tags": [],
    "tenant_id": "e153f3f9082240a5974f667cfe1036e3",
    "updated_at": "2018-10-03T13:57:33Z"
  }
}`

const ListSubportsResponse = `
{
  "sub_ports": [
    {
      "port_id": "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
      "segmentation_id": 1,
      "segmentation_type": "vlan"
    },
    {
      "port_id": "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
      "segmentation_id": 2,
      "segmentation_type": "vlan"
    }
  ]
}`

const AddSubportsRequest = ListSubportsResponse

const AddSubportsResponse = `
{
  "admin_state_up": true,
  "created_at": "2018-10-03T13:57:24Z",
  "description": "Trunk created by gophercloud",
  "id": "f6a9718c-5a64-43e3-944f-4deccad8e78c",
  "name": "gophertrunk",
  "port_id": "c373d2fa-3d3b-4492-924c-aff54dea19b6",
  "project_id": "e153f3f9082240a5974f667cfe1036e3",
  "revision_number": 2,
  "status": "ACTIVE",
  "sub_ports": [
    {
      "port_id": "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
      "segmentation_id": 1,
      "segmentation_type": "vlan"
    },
    {
      "port_id": "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
      "segmentation_id": 2,
      "segmentation_type": "vlan"
    }
  ],
  "tags": [],
  "tenant_id": "e153f3f9082240a5974f667cfe1036e3",
  "updated_at": "2018-10-03T13:57:30Z"
}`

const RemoveSubportsRequest = `
{
  "sub_ports": [
    {
      "port_id": "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b"
    },
    {
      "port_id": "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab"
    }
  ]
}`

const RemoveSubportsResponse = `
{
  "admin_state_up": true,
  "created_at": "2018-10-03T13:57:24Z",
  "description": "Trunk created by gophercloud",
  "id": "f6a9718c-5a64-43e3-944f-4deccad8e78c",
  "name": "gophertrunk",
  "port_id": "c373d2fa-3d3b-4492-924c-aff54dea19b6",
  "project_id": "e153f3f9082240a5974f667cfe1036e3",
  "revision_number": 2,
  "status": "ACTIVE",
  "sub_ports": [],
  "tags": [],
  "tenant_id": "e153f3f9082240a5974f667cfe1036e3",
  "updated_at": "2018-10-03T13:57:27Z"
}`

var ExpectedSubports = []trunks.Subport{
	{
		PortID:           "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
		SegmentationID:   1,
		SegmentationType: "vlan",
	},
	{
		PortID:           "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
		SegmentationID:   2,
		SegmentationType: "vlan",
	},
}

func ExpectedTrunkSlice() (exp []trunks.Trunk, err error) {
	trunk1CreatedAt, err := time.Parse(time.RFC3339, "2018-10-01T15:29:39Z")
	if err != nil {
		return nil, err
	}

	trunk1UpdatedAt, err := time.Parse(time.RFC3339, "2018-10-01T15:43:04Z")
	if err != nil {
		return nil, err
	}
	exp = make([]trunks.Trunk, 2)
	exp[0] = trunks.Trunk{
		AdminStateUp:   true,
		Description:    "",
		ID:             "3e72aa1b-d0da-48f2-831a-fd1c5f3f99c2",
		Name:           "mytrunk",
		PortID:         "16c425d3-d7fc-40b8-b94c-cc95da45b270",
		ProjectID:      "e153f3f9082240a5974f667cfe1036e3",
		TenantID:       "e153f3f9082240a5974f667cfe1036e3",
		RevisionNumber: 3,
		Status:         "ACTIVE",
		Subports: []trunks.Subport{
			{
				PortID:           "424da4b7-7868-4db2-bb71-05155601c6e4",
				SegmentationID:   11,
				SegmentationType: "vlan",
			},
		},
		Tags:      []string{},
		CreatedAt: trunk1CreatedAt,
		UpdatedAt: trunk1UpdatedAt,
	}

	trunk2CreatedAt, err := time.Parse(time.RFC3339, "2018-10-03T13:57:24Z")
	if err != nil {
		return nil, err
	}

	trunk2UpdatedAt, err := time.Parse(time.RFC3339, "2018-10-03T13:57:26Z")
	if err != nil {
		return nil, err
	}
	exp[1] = trunks.Trunk{
		AdminStateUp:   true,
		Description:    "Trunk created by gophercloud",
		ID:             "f6a9718c-5a64-43e3-944f-4deccad8e78c",
		Name:           "gophertrunk",
		PortID:         "c373d2fa-3d3b-4492-924c-aff54dea19b6",
		ProjectID:      "e153f3f9082240a5974f667cfe1036e3",
		TenantID:       "e153f3f9082240a5974f667cfe1036e3",
		RevisionNumber: 1,
		Status:         "ACTIVE",
		Subports:       ExpectedSubports,
		Tags:           []string{},
		CreatedAt:      trunk2CreatedAt,
		UpdatedAt:      trunk2UpdatedAt,
	}
	return
}

func ExpectedSubportsAddedTrunk() (exp trunks.Trunk, err error) {
	trunkUpdatedAt, err := time.Parse(time.RFC3339, "2018-10-03T13:57:30Z")
	expectedTrunks, err := ExpectedTrunkSlice()
	if err != nil {
		return
	}
	exp = expectedTrunks[1]
	exp.RevisionNumber += 1
	exp.UpdatedAt = trunkUpdatedAt
	return
}

func ExpectedSubportsRemovedTrunk() (exp trunks.Trunk, err error) {
	trunkUpdatedAt, err := time.Parse(time.RFC3339, "2018-10-03T13:57:27Z")
	expectedTrunks, err := ExpectedTrunkSlice()
	if err != nil {
		return
	}
	exp = expectedTrunks[1]
	exp.RevisionNumber += 1
	exp.UpdatedAt = trunkUpdatedAt
	exp.Subports = []trunks.Subport{}
	return
}
