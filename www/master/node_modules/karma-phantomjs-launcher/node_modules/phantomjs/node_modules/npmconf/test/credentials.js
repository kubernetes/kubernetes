var test = require("tap").test

var npmconf = require("../npmconf.js")
var common = require("./00-setup.js")

var URI = "https://registry.lvh.me:8661/"

test("getting scope with no credentials set", function (t) {
  npmconf.load({}, function (er, conf) {
    t.ifError(er, "configuration loaded")

    var basic = conf.getCredentialsByURI(URI)
    t.equal(basic.scope, "//registry.lvh.me:8661/", "nerfed URL extracted")

    t.end()
  })
})

test("trying to set credentials with no URI", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    t.throws(function () {
      conf.setCredentialsByURI()
    }, "enforced missing URI")

    t.end()
  })
})

test("set with missing credentials object", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    t.throws(function () {
      conf.setCredentialsByURI(URI)
    }, "enforced missing credentials")

    t.end()
  })
})

test("set with empty credentials object", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    t.throws(function () {
      conf.setCredentialsByURI(URI, {})
    }, "enforced missing credentials")

    t.end()
  })
})

test("set with token", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    t.doesNotThrow(function () {
      conf.setCredentialsByURI(URI, {token : "simple-token"})
    }, "needs only token")

    var expected = {
      scope      : "//registry.lvh.me:8661/",
      token      : "simple-token",
      username   : undefined,
      password   : undefined,
      email      : undefined,
      auth       : undefined,
      alwaysAuth : undefined
    }

    t.same(conf.getCredentialsByURI(URI), expected, "got bearer token and scope")

    t.end()
  })
})

test("set with missing username", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    var credentials = {
      password : "password",
      email    : "ogd@aoaioxxysz.net"
    }

    t.throws(function () {
      conf.setCredentialsByURI(URI, credentials)
    }, "enforced missing email")

    t.end()
  })
})

test("set with missing password", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    var credentials = {
      username : "username",
      email    : "ogd@aoaioxxysz.net"
    }

    t.throws(function () {
      conf.setCredentialsByURI(URI, credentials)
    }, "enforced missing email")

    t.end()
  })
})

test("set with missing email", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    var credentials = {
      username : "username",
      password : "password"
    }

    t.throws(function () {
      conf.setCredentialsByURI(URI, credentials)
    }, "enforced missing email")

    t.end()
  })
})

test("set with old-style credentials", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    var credentials = {
      username : "username",
      password : "password",
      email    : "ogd@aoaioxxysz.net"
    }

    t.doesNotThrow(function () {
      conf.setCredentialsByURI(URI, credentials)
    }, "requires all of username, password, and email")

    var expected = {
      scope      : "//registry.lvh.me:8661/",
      token      : undefined,
      username   : "username",
      password   : "password",
      email      : "ogd@aoaioxxysz.net",
      auth       : "dXNlcm5hbWU6cGFzc3dvcmQ=",
      alwaysAuth : false
    }

    t.same(conf.getCredentialsByURI(URI), expected, "got credentials")

    t.end()
  })
})

test("get old-style credentials for default registry", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    var actual = conf.getCredentialsByURI(conf.get("registry"))
    var expected = {
      scope      : '//registry.npmjs.org/',
      token      : undefined,
      password   : 'password',
      username   : 'username',
      email      : 'i@izs.me',
      auth       : 'dXNlcm5hbWU6cGFzc3dvcmQ=',
      alwaysAuth : false
    }
    t.same(actual, expected)
    t.end()
  })
})

test("set with always-auth enabled", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    var credentials = {
      username   : "username",
      password   : "password",
      email      : "ogd@aoaioxxysz.net",
      alwaysAuth : true
    }

    conf.setCredentialsByURI(URI, credentials)

    var expected = {
      scope      : "//registry.lvh.me:8661/",
      token      : undefined,
      username   : "username",
      password   : "password",
      email      : "ogd@aoaioxxysz.net",
      auth       : "dXNlcm5hbWU6cGFzc3dvcmQ=",
      alwaysAuth : true
    }

    t.same(conf.getCredentialsByURI(URI), expected, "got credentials")

    t.end()
  })
})

test("set with always-auth disabled", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")

    var credentials = {
      username   : "username",
      password   : "password",
      email      : "ogd@aoaioxxysz.net",
      alwaysAuth : false
    }

    conf.setCredentialsByURI(URI, credentials)

    var expected = {
      scope      : "//registry.lvh.me:8661/",
      token      : undefined,
      username   : "username",
      password   : "password",
      email      : "ogd@aoaioxxysz.net",
      auth       : "dXNlcm5hbWU6cGFzc3dvcmQ=",
      alwaysAuth : false
    }

    t.same(conf.getCredentialsByURI(URI), expected, "got credentials")

    t.end()
  })
})

test("set with global always-auth enabled", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")
    var original = conf.get("always-auth")
    conf.set("always-auth", true)

    var credentials = {
      username   : "username",
      password   : "password",
      email      : "ogd@aoaioxxysz.net"
    }

    conf.setCredentialsByURI(URI, credentials)

    var expected = {
      scope      : "//registry.lvh.me:8661/",
      token      : undefined,
      username   : "username",
      password   : "password",
      email      : "ogd@aoaioxxysz.net",
      auth       : "dXNlcm5hbWU6cGFzc3dvcmQ=",
      alwaysAuth : true
    }

    t.same(conf.getCredentialsByURI(URI), expected, "got credentials")

    conf.set("always-auth", original)
    t.end()
  })
})

test("set with global always-auth disabled", function (t) {
  npmconf.load(common.builtin, function (er, conf) {
    t.ifError(er, "configuration loaded")
    var original = conf.get("always-auth")
    conf.set("always-auth", false)

    var credentials = {
      username   : "username",
      password   : "password",
      email      : "ogd@aoaioxxysz.net"
    }

    conf.setCredentialsByURI(URI, credentials)

    var expected = {
      scope      : "//registry.lvh.me:8661/",
      token      : undefined,
      username   : "username",
      password   : "password",
      email      : "ogd@aoaioxxysz.net",
      auth       : "dXNlcm5hbWU6cGFzc3dvcmQ=",
      alwaysAuth : false
    }

    t.same(conf.getCredentialsByURI(URI), expected, "got credentials")

    conf.set("always-auth", original)
    t.end()
  })
})
