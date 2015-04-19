var assert = require("assert")

var toNerfDart = require("./nerf-dart.js")

module.exports = setCredentialsByURI

function setCredentialsByURI (uri, c) {
  assert(uri && typeof uri === "string", "registry URL is required")
  assert(c && typeof c === "object", "credentials are required")

  var nerfed = toNerfDart(uri)

  if (c.token) {
    this.set(nerfed + ":_authToken", c.token, "user")
    this.del(nerfed + ":_password",           "user")
    this.del(nerfed + ":username",            "user")
    this.del(nerfed + ":email",               "user")
    this.del(nerfed + ":always-auth",         "user")
  }
  else if (c.username || c.password || c.email) {
    assert(c.username, "must include username")
    assert(c.password, "must include password")
    assert(c.email, "must include email address")

    this.del(nerfed + ":_authToken", "user")

    var encoded = new Buffer(c.password, "utf8").toString("base64")
    this.set(nerfed + ":_password", encoded,   "user")
    this.set(nerfed + ":username", c.username, "user")
    this.set(nerfed + ":email", c.email,       "user")

    if (c.alwaysAuth !== undefined) {
      this.set(nerfed + ":always-auth", c.alwaysAuth, "user")
    }
    else {
      this.del(nerfed + ":always-auth", "user")
    }
  }
  else {
    throw new Error("No credentials to set.")
  }
}
