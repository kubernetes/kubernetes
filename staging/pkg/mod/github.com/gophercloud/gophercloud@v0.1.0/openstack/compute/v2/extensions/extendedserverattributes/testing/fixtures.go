package testing

// ServerWithAttributesExtResult represents a raw server response from the
// Compute API with OS-EXT-SRV-ATTR data.
// Most of the actual fields were deleted from the response.
const ServerWithAttributesExtResult = `
{
    "server": {
        "OS-EXT-SRV-ATTR:user_data": "",
        "OS-EXT-SRV-ATTR:instance_name": "instance-00000001",
        "OS-EXT-SRV-ATTR:root_device_name": "/dev/sda",
        "OS-EXT-SRV-ATTR:hostname": "test00",
        "OS-EXT-SRV-ATTR:reservation_id": "r-ky9gim1l",
        "OS-EXT-SRV-ATTR:ramdisk_id": "",
        "OS-EXT-SRV-ATTR:host": "compute01",
        "OS-EXT-SRV-ATTR:kernel_id": "",
        "OS-EXT-SRV-ATTR:hypervisor_hostname": "compute01",
        "OS-EXT-SRV-ATTR:launch_index": 0,
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
