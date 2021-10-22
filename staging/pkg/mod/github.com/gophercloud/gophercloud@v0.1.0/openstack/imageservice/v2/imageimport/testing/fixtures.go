package testing

// ImportGetResult represents raw server response on a Get request.
const ImportGetResult = `
{
    "import-methods": {
        "description": "Import methods available.",
        "type": "array",
        "value": [
            "glance-direct",
            "web-download"
        ]
    }
}
`

// ImportCreateRequest represents a request to create image import.
const ImportCreateRequest = `
{
    "method": {
        "name": "web-download",
        "uri": "http://download.cirros-cloud.net/0.4.0/cirros-0.4.0-x86_64-disk.img"
    }
}
`
