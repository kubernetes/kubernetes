package testing

import (
	"time"

	"github.com/gophercloud/gophercloud/openstack/imageservice/v2/tasks"
)

// TasksListResult represents raw server response from a server to a list call.
const TasksListResult = `
{
    "schema": "/v2/schemas/tasks",
    "tasks": [
        {
            "status": "pending",
            "self": "/v2/tasks/1252f636-1246-4319-bfba-c47cde0efbe0",
            "updated_at": "2018-07-25T08:59:14Z",
            "id": "1252f636-1246-4319-bfba-c47cde0efbe0",
            "owner": "424e7cf0243c468ca61732ba45973b3e",
            "type": "import",
            "created_at": "2018-07-25T08:59:13Z",
            "schema": "/v2/schemas/task"
        },
        {
            "status": "processing",
            "self": "/v2/tasks/349a51f4-d51d-47b6-82da-4fa516f0ca32",
            "updated_at": "2018-07-25T08:56:19Z",
            "id": "349a51f4-d51d-47b6-82da-4fa516f0ca32",
            "owner": "fb57277ef2f84a0e85b9018ec2dedbf7",
            "type": "import",
            "created_at": "2018-07-25T08:56:17Z",
            "schema": "/v2/schemas/task"
        }
    ],
    "first": "/v2/tasks?sort_key=status&sort_dir=desc&limit=20"
}
`

// Task1 is an expected representation of a first task from the TasksListResult.
var Task1 = tasks.Task{
	ID:        "1252f636-1246-4319-bfba-c47cde0efbe0",
	Status:    string(tasks.TaskStatusPending),
	Type:      "import",
	Owner:     "424e7cf0243c468ca61732ba45973b3e",
	CreatedAt: time.Date(2018, 7, 25, 8, 59, 13, 0, time.UTC),
	UpdatedAt: time.Date(2018, 7, 25, 8, 59, 14, 0, time.UTC),
	Self:      "/v2/tasks/1252f636-1246-4319-bfba-c47cde0efbe0",
	Schema:    "/v2/schemas/task",
}

// Task2 is an expected representation of a first task from the TasksListResult.
var Task2 = tasks.Task{
	ID:        "349a51f4-d51d-47b6-82da-4fa516f0ca32",
	Status:    string(tasks.TaskStatusProcessing),
	Type:      "import",
	Owner:     "fb57277ef2f84a0e85b9018ec2dedbf7",
	CreatedAt: time.Date(2018, 7, 25, 8, 56, 17, 0, time.UTC),
	UpdatedAt: time.Date(2018, 7, 25, 8, 56, 19, 0, time.UTC),
	Self:      "/v2/tasks/349a51f4-d51d-47b6-82da-4fa516f0ca32",
	Schema:    "/v2/schemas/task",
}

// TasksGetResult represents raw server response from a server to a get call.
const TasksGetResult = `
{
    "status": "pending",
    "created_at": "2018-07-25T08:59:13Z",
    "updated_at": "2018-07-25T08:59:14Z",
    "self": "/v2/tasks/1252f636-1246-4319-bfba-c47cde0efbe0",
    "result": null,
    "owner": "424e7cf0243c468ca61732ba45973b3e",
    "input": {
        "image_properties": {
            "container_format": "bare",
            "disk_format": "raw"
        },
        "import_from_format": "raw",
        "import_from": "http://download.cirros-cloud.net/0.4.0/cirros-0.4.0-x86_64-disk.img"
    },
    "message": "",
    "type": "import",
    "id": "1252f636-1246-4319-bfba-c47cde0efbe0",
    "schema": "/v2/schemas/task"
}
`

// TaskCreateRequest represents a request to create a task.
const TaskCreateRequest = `
{
    "input": {
        "image_properties": {
            "container_format": "bare",
            "disk_format": "raw"
        },
        "import_from_format": "raw",
        "import_from": "https://cloud-images.ubuntu.com/bionic/current/bionic-server-cloudimg-amd64.img"
    },
    "type": "import"
}
`

// TaskCreateResult represents a raw server response to the TaskCreateRequest.
const TaskCreateResult = `
{
    "status": "pending",
    "created_at": "2018-07-25T11:07:54Z",
    "updated_at": "2018-07-25T11:07:54Z",
    "self": "/v2/tasks/d550c87d-86ed-430a-9895-c7a1f5ce87e9",
    "result": null,
    "owner": "fb57277ef2f84a0e85b9018ec2dedbf7",
    "input": {
        "image_properties": {
            "container_format": "bare",
            "disk_format": "raw"
        },
        "import_from_format": "raw",
        "import_from": "https://cloud-images.ubuntu.com/bionic/current/bionic-server-cloudimg-amd64.img"
    },
    "message": "",
    "type": "import",
    "id": "d550c87d-86ed-430a-9895-c7a1f5ce87e9",
    "schema": "/v2/schemas/task"
}
`
