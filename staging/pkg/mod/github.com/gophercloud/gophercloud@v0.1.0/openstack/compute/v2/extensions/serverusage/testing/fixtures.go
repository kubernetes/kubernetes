package testing

// ServerWithUsageExtResult represents a raw server response from the Compute API
// with OS-SRV-USG data.
// Most of the actual fields were deleted from the response.
const ServerWithUsageExtResult = `
{
    "server": {
        "OS-SRV-USG:launched_at": "2018-07-27T09:15:55.000000",
        "OS-SRV-USG:terminated_at": null,
        "created": "2018-07-27T09:15:48Z",
        "updated": "2018-07-27T09:15:55Z",
        "id": "d650a0ce-17c3-497d-961a-43c4af80998a",
        "name": "test_instance",
        "status": "ACTIVE",
        "user_id": "0f2f3822679e4b3ea073e5d1c6ed5f02",
        "tenant_id": "424e7cf0243c468ca61732ba45973b3e"
    }
}
`
