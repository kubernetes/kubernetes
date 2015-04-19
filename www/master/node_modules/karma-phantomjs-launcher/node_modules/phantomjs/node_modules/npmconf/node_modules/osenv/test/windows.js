// only run this test on windows
// pretending to be another platform is too hacky, since it breaks
// how the underlying system looks up module paths and runs
// child processes, and all that stuff is cached.
if (process.platform !== 'win32') {
  console.log('TAP Version 13\n' +
              '1..0\n' +
              '# Skip windows tests, this is not windows\n')
  return
}

// load this before clubbing the platform name.
var tap = require('tap')

process.env.windir = 'c:\\windows'
process.env.USERDOMAIN = 'some-domain'
process.env.USERNAME = 'sirUser'
process.env.USERPROFILE = 'C:\\Users\\sirUser'
process.env.COMPUTERNAME = 'my-machine'
process.env.TMPDIR = 'C:\\tmpdir'
process.env.TMP = 'C:\\tmp'
process.env.TEMP = 'C:\\temp'
process.env.Path = 'C:\\Program Files\\;C:\\Binary Stuff\\bin'
process.env.PROMPT = '(o_o) $ '
process.env.EDITOR = 'edit'
process.env.VISUAL = 'visualedit'
process.env.ComSpec = 'some-com'

tap.test('basic windows sanity test', function (t) {
  var osenv = require('../osenv.js')

  t.equal(osenv.user(),
          process.env.USERDOMAIN + '\\' + process.env.USERNAME)
  t.equal(osenv.home(), process.env.USERPROFILE)
  t.equal(osenv.hostname(), process.env.COMPUTERNAME)
  t.same(osenv.path(), process.env.Path.split(';'))
  t.equal(osenv.prompt(), process.env.PROMPT)
  t.equal(osenv.tmpdir(), process.env.TMPDIR)

  // mildly evil, but it's for a test.
  process.env.TMPDIR = ''
  delete require.cache[require.resolve('../osenv.js')]
  var osenv = require('../osenv.js')
  t.equal(osenv.tmpdir(), process.env.TMP)

  process.env.TMP = ''
  delete require.cache[require.resolve('../osenv.js')]
  var osenv = require('../osenv.js')
  t.equal(osenv.tmpdir(), process.env.TEMP)

  process.env.TEMP = ''
  delete require.cache[require.resolve('../osenv.js')]
  var osenv = require('../osenv.js')
  osenv.home = function () { return null }
  t.equal(osenv.tmpdir(), 'c:\\windows\\temp')

  t.equal(osenv.editor(), 'edit')
  process.env.EDITOR = ''
  delete require.cache[require.resolve('../osenv.js')]
  var osenv = require('../osenv.js')
  t.equal(osenv.editor(), 'visualedit')

  process.env.VISUAL = ''
  delete require.cache[require.resolve('../osenv.js')]
  var osenv = require('../osenv.js')
  t.equal(osenv.editor(), 'notepad.exe')

  t.equal(osenv.shell(), 'some-com')
  process.env.ComSpec = ''
  delete require.cache[require.resolve('../osenv.js')]
  var osenv = require('../osenv.js')
  t.equal(osenv.shell(), 'cmd')

  t.end()
})
