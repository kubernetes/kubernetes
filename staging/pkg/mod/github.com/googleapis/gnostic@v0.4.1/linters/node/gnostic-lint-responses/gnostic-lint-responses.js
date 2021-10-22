// import libraries
const protobuf = require("protobufjs");
const getStdin = require("get-stdin");
const find = require("lodash/find");
const forEach = require("lodash/forEach");
const pick = require("lodash/pick");

// import messages
const root = protobuf.Root.fromJSON(require("./bundle.json"));
const Request = root.lookupType("gnostic.plugin.v1.Request");
const Response = root.lookupType("gnostic.plugin.v1.Response");
const Document = root.lookupType("openapi.v2.Document");

getStdin.buffer().then(buffer => {
  const request = Request.decode(buffer);
  var messages = [];
  for (var j in request.models) {
    const m = request.models[j];
    if (m.type_url == "openapi.v2.Document") {
      const openapi2 = Document.decode(m.value);
      const paths = openapi2.paths.path;
      for (var i in paths) {
        const path = paths[i];
        // console.error('path %s\n\n', path.name)

        // Arrays MUST NOT be returned as the top-level structure in a response body.
        let pathOps = pick(path.value, ["get","head","post", "put", "patch", "delete", "options"]);
        forEach(pathOps, (op, opKey) => {
          if (op != null) {
            forEach(op.responses.responseCode, responseObj => {
              // console.error('responseObj is %j', responseObj)
              name = responseObj.name;
              response = responseObj.value.response;
              if (response.schema && response.schema.schema) {
                if (!response.schema.schema._ref) {
                  if (
                    response.schema.schema.type != null &&
                    response.schema.schema.type.value == "array"
                  ) {
                    messages.push({
                      level: 3,
                      code: "NO_ARRAY_RESPONSES",
                      text: "Arrays MUST NOT be returned as the top-level structure in a response body.",
                      keys: ["paths", path.name, opKey, "responses", name, "schema"]
                    });
                  }
                } else {
                  let schemaName = response.schema.schema._ref.match(/#\/definitions\/(\w+)/);
                  if (schemaName) {
                    const definitions = openapi2.definitions.additionalProperties;
                    const schemaKvp = find(definitions, {name: schemaName[1]
                    });
                    //console.error('schemaKvp.value.type = %s', schemaKvp.value.type.value)
                    if (schemaKvp && schemaKvp.value.type && schemaKvp.value.type.value.indexOf("array") >= 0) {
                      messages.push({
                        level: 3,
                        code: "NO_ARRAY_RESPONSES",
                        text: "Arrays MUST NOT be returned as the top-level structure in a response body.",
                        keys: ["paths", path.name, opKey, "responses", name, "schema" ]
                      });
                    }
                  }
                }
              }
            });
          }
        });
      }
    }
  }

  const payload = {
    messages: messages
  };

  // Verify the payload if necessary (i.e. when possibly incomplete or invalid)
  const errMsg = Response.verify(payload);
  if (errMsg) throw Error(errMsg);

  const message = Response.create(payload);
  process.stdout.write(Response.encode(message).finish());
})
.catch(err => console.error(err));
