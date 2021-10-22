package testing

// RescueRequest represents request to rescue a server.
const RescueRequest = `
{
    "rescue": {
        "adminPass": "aUPtawPzE9NU",
        "rescue_image_ref": "115e5c5b-72f0-4a0a-9067-60706545248c"
    }
}
`

// RescueResult represents a raw server response to a RescueRequest.
const RescueResult = `
{
	"adminPass": "aUPtawPzE9NU"
}
`

// UnrescueRequest represents request to unrescue a server.
const UnrescueRequest = `
{
    "unrescue": null
}
`
