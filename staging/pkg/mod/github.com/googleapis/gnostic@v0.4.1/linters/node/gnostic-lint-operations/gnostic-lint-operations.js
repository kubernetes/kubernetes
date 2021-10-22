// import libraries
const protobuf = require("protobufjs")
const getStdin = require('get-stdin')

// import messages
const root = protobuf.Root.fromJSON(require("./bundle.json"))
const Request = root.lookupType("gnostic.plugin.v1.Request")
const Response = root.lookupType("gnostic.plugin.v1.Response")
const Document = root.lookupType("openapi.v2.Document")

getStdin.buffer().then(buffer => {
	const request = Request.decode(buffer)
	messages = []
	for (var j in request.models) {
		const m = request.models[j]
		if (m.type_url == "openapi.v2.Document") {
			const openapi2 = Document.decode(m.value)
			const paths = openapi2.paths.path
			for (var i in paths) {
				const path = paths[i]
				//console.error('path %s\n\n', path.name)
				const getOperation = path.value.get
				if (getOperation && getOperation.operationId == "") {
					messages.push({level:3, code:"NOOPERATIONID", text:"No operation id.", keys:["paths", path.name, "get"]})
				}
				const postOperation = path.value.post
				if (postOperation && postOperation.operationId == "") {
					messages.push({level:3, code:"NOOPERATIONID", text:"No operation id.", keys:["paths", path.name, "post"]})
				}
				//console.error('get %s\n\n', JSON.stringify(getOperation))
			}
		}
	}
	
	const payload = {
		messages: messages
	}

	// Verify the payload if necessary (i.e. when possibly incomplete or invalid)
	const errMsg = Response.verify(payload)
	if (errMsg)
		throw Error(errMsg)

	const message = Response.create(payload)
	process.stdout.write(Response.encode(message).finish())

}).catch(err => console.error(err))
