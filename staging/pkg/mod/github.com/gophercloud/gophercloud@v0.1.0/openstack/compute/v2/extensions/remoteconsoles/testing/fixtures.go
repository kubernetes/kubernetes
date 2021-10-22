package testing

// RemoteConsoleCreateRequest represents a request to create a remote console.
const RemoteConsoleCreateRequest = `
{
    "remote_console": {
        "protocol": "vnc",
        "type": "novnc"
    }
}
`

// RemoteConsoleCreateResult represents a raw server responce to the RemoteConsoleCreateRequest.
const RemoteConsoleCreateResult = `
{
    "remote_console": {
        "protocol": "vnc",
        "type": "novnc",
        "url": "http://192.168.0.4:6080/vnc_auto.html?token=9a2372b9-6a0e-4f71-aca1-56020e6bb677"
    }
}
`
