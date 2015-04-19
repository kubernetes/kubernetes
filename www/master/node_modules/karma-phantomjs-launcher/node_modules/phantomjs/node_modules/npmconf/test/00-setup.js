var path = require('path')
var userconfigSrc = path.resolve(__dirname, 'fixtures', 'userconfig')
exports.userconfig = userconfigSrc + '-with-gc'
exports.globalconfig = path.resolve(__dirname, 'fixtures', 'globalconfig')
exports.builtin = path.resolve(__dirname, 'fixtures', 'builtin')
exports.ucData =
  { globalconfig: exports.globalconfig,
    email: 'i@izs.me',
    'env-thing': 'asdf',
    'init.author.name': 'Isaac Z. Schlueter',
    'init.author.email': 'i@izs.me',
    'init.author.url': 'http://blog.izs.me/',
    'init.version': '1.2.3',
    'proprietary-attribs': false,
    'npm:publishtest': true,
    '_npmjs.org:couch': 'https://admin:password@localhost:5984/registry',
    'npm-www:nocache': '1',
    nodedir: '/Users/isaacs/dev/js/node-v0.8',
    'sign-git-tag': true,
    message: 'v%s',
    'strict-ssl': false,
    'tmp': process.env.HOME + '/.tmp',
    _auth: 'dXNlcm5hbWU6cGFzc3dvcmQ=',
    _token:
     { AuthSession: 'yabba-dabba-doodle',
       version: '1',
       expires: '1345001053415',
       path: '/',
       httponly: true } }

// set the userconfig in the env
// unset anything else that npm might be trying to foist on us
Object.keys(process.env).forEach(function (k) {
  if (k.match(/^npm_config_/i)) {
    delete process.env[k]
  }
})
process.env.npm_config_userconfig = exports.userconfig
process.env.npm_config_other_env_thing = 1000
process.env.random_env_var = 'asdf'
process.env.npm_config__underbar_env_thing = 'underful'
process.env.NPM_CONFIG_UPPERCASE_ENV_THING = 42

exports.envData = {
  userconfig: exports.userconfig,
  '_underbar-env-thing': 'underful',
  'uppercase-env-thing': '42',
  'other-env-thing': '1000'
}
exports.envDataFix = {
  userconfig: exports.userconfig,
  '_underbar-env-thing': 'underful',
  'uppercase-env-thing': 42,
  'other-env-thing': 1000,
}


if (module === require.main) {
  // set the globalconfig in the userconfig
  var fs = require('fs')
  var uc = fs.readFileSync(userconfigSrc)
  var gcini = 'globalconfig = ' + exports.globalconfig + '\n'
  fs.writeFileSync(exports.userconfig, gcini + uc)

  console.log('0..1')
  console.log('ok 1 setup done')
}
