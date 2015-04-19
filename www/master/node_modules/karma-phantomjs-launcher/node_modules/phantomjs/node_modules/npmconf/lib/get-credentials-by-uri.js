var assert = require("assert")

var toNerfDart = require("./nerf-dart.js")

module.exports = getCredentialsByURI

function getCredentialsByURI (uri) {
  assert(uri && typeof uri === "string", "registry URL is required")
  var nerfed = toNerfDart(uri)
  var defnerf = toNerfDart(this.get("registry"))

  // hidden class micro-optimization
  var c = {
    scope      : nerfed,
    token      : undefined,
    password   : undefined,
    username   : undefined,
    email      : undefined,
    auth       : undefined,
    alwaysAuth : undefined
  }

  if (this.get(nerfed + ":_authToken")) {
    c.token = this.get(nerfed + ":_authToken")
    // the bearer token is enough, don't confuse things
    return c
  }

  // Handle the old-style _auth=<base64> style for the default
  // registry, if set.
  //
  // XXX(isaacs): Remove when npm 1.4 is no longer relevant
  var authDef = this.get("_auth")
  var userDef = this.get("username")
  var passDef = this.get("_password")
  if (authDef && !(userDef && passDef)) {
    authDef = new Buffer(authDef, "base64").toString()
    authDef = authDef.split(":")
    userDef = authDef.shift()
    passDef = authDef.join(":")
  }

  if (this.get(nerfed + ":_password")) {
    c.password = new Buffer(this.get(nerfed + ":_password"), "base64").toString("utf8")
  } else if (nerfed === defnerf && passDef) {
    c.password = passDef
  }

  if (this.get(nerfed + ":username")) {
    c.username = this.get(nerfed + ":username")
  } else if (nerfed === defnerf && userDef) {
    c.username = userDef
  }

  if (this.get(nerfed + ":email")) {
    c.email = this.get(nerfed + ":email")
  } else if (this.get("email")) {
    c.email = this.get("email")
  }

  if (this.get(nerfed + ":always-auth") !== undefined) {
    var val = this.get(nerfed + ":always-auth")
    c.alwaysAuth = val === "false" ? false : !!val
  } else if (this.get("always-auth") !== undefined) {
    c.alwaysAuth = this.get("always-auth")
  }

  if (c.username && c.password) {
    c.auth = new Buffer(c.username + ":" + c.password).toString("base64")
  }

  return c
}
